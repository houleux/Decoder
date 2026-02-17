#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <numeric>
#include <algorithm>

// Utility for Box-Plus operation (Check Node Update)
// Corresponds to Equation (2) and (3) in the paper [cite: 81, 82]
double box_plus(double L1, double L2) {
    // 2 * atanh( tanh(L1/2) * tanh(L2/2) )
    // Clamping to avoid numerical instability with atanh(1) or atanh(-1)
    double t1 = std::tanh(L1 / 2.0);
    double t2 = std::tanh(L2 / 2.0);
    double prod = t1 * t2;
    
    if (prod >= 1.0) prod = 0.9999999999;
    if (prod <= -1.0) prod = -0.9999999999;
    
    return 2.0 * std::atanh(prod);
}

class RBLBP_Decoder {
private:
    int N; // Number of variable nodes (Code length)
    int M; // Number of check nodes
    std::vector<std::vector<int>> H_rows; // Adjacency list: Check -> Variables
    std::vector<std::vector<int>> H_cols; // Adjacency list: Variable -> Checks
    
    // Message matrices
    // R[m][n] stores message from Check m -> Variable n
    // Q[m][n] stores message from Variable n -> Check m
    // Using dense matrices for simplicity; for large codes, use sparse structures/maps.
    std::vector<std::vector<double>> R;
    std::vector<std::vector<double>> Q;

public:
    // Constructor to initialize Tanner Graph from Parity Check Matrix
    RBLBP_Decoder(int rows, int cols, const std::vector<std::vector<int>>& parity_matrix) 
        : M(rows), N(cols) {
        
        H_rows.resize(M);
        H_cols.resize(N);
        
        // Build adjacency lists
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                if (parity_matrix[m][n] == 1) {
                    H_rows[m].push_back(n);
                    H_cols[n].push_back(m);
                }
            }
        }
        
        // Resize message containers
        R.resize(M, std::vector<double>(N, 0.0));
        Q.resize(M, std::vector<double>(N, 0.0));
    }

    // Main RBL-BP Decoding Function
    // Implements logic from Table III 
    std::vector<int> decode(const std::vector<double>& received_LLR, int max_iter, double alpha) {
        
        // --- Step 1 & 2: Initialization [cite: 126] ---
        // Initialize R to 0
        for(auto& row : R) std::fill(row.begin(), row.end(), 0.0);
        
        // Initialize Q to Channel LLR
        for (int m = 0; m < M; ++m) {
            for (int n : H_rows[m]) {
                Q[m][n] = received_LLR[n];
            }
        }

        // --- Step 3: Find initial threshold [cite: 127-130] ---
        double min_abs_llr = std::numeric_limits<double>::max();
        for (double val : received_LLR) {
            if (std::abs(val) < min_abs_llr) {
                min_abs_llr = std::abs(val);
            }
        }
        
        double L_value_threshold = min_abs_llr;
        std::vector<int> decoded_bits(N);

        // --- Iterative Loop ---
        for (int l = 1; l <= max_iter; ++l) {
            
            // --- Step 4: Identify Active Variables [cite: 134] ---
            // Find all n where |Ln(0)| <= L_value_threshold
            std::vector<int> active_vars;
            for (int n = 0; n < N; ++n) {
                if (std::abs(received_LLR[n]) <= L_value_threshold) {
                    active_vars.push_back(n);
                }
            }

            // --- Steps 5-11: Layered Update Schedule [cite: 146-161] ---
            // The paper implies updating neighbors of the "unreliable" nodes first.
            
            for (int n : active_vars) {
                
                // For each Check Node 'k' connected to active Variable 'n' [cite: 146]
                for (int k : H_cols[n]) {
                    
                    // --- Generate and propagate Check-to-Variable messages (R) ---
                    // For check k, update R messages to all its neighbors 'a'
                    // Corresponds to Step 6 [cite: 136, 148]
                    
                    // Pre-calculate total product (box-sum) for efficiency if possible, 
                    // or just loop normally for clarity.
                    for (int a : H_rows[k]) {
                        double product_msg = 1.0; // Identity for tanh product
                        bool first = true;
                        
                        // Product of tanh(Q/2) for all n' in N(k) \ a [cite: 85]
                        for (int n_prime : H_rows[k]) {
                            if (n_prime != a) {
                                // Clamp tanh to avoid 1.0/-1.0
                                double t = std::tanh(Q[k][n_prime] / 2.0);
                                product_msg *= t;
                            }
                        }
                        
                        // Update R message [cite: 148]
                        // Clamp product to avoid atanh domain errors
                        if(product_msg > 0.999999) product_msg = 0.999999;
                        if(product_msg < -0.999999) product_msg = -0.999999;
                        
                        R[k][a] = 2.0 * std::atanh(product_msg);
                    }

                    // --- Generate and propagate Variable-to-Check messages (Q) ---
                    // After updating check k, we immediately update the outgoing Q messages
                    // for the variable nodes 'a' connected to k.
                    // Corresponds to Step 9 [cite: 149, 140]
                    
                    for (int a : H_rows[k]) { // For variables 'a' connected to check 'k'
                        // Update Q from 'a' to all its OTHER check neighbors 'b'
                        // Formula: L(0) + sum(R from others) [cite: 92]
                        
                        for (int b : H_cols[a]) {
                            if (b == k) continue; // Only propagate to others? 
                            // *Note:* Standard LBP usually updates ALL outgoing edges from 'a' 
                            // using the newest info from 'k'.
                            
                            double sum_R = 0.0;
                            for (int m_prime : H_cols[a]) {
                                if (m_prime != b) {
                                    sum_R += R[m_prime][a];
                                }
                            }
                            Q[b][a] = received_LLR[a] + sum_R; 
                        }
                        
                        // Also need to update Q[k][a] for the next time k is used? 
                        // The paper says "For b in M(a)\k"[cite: 149], but strictly speaking,
                        // to keep state consistent for other checks, we update the node 'a'.
                        // We strictly follow the loop: for b in M(a)\k, update Q_{b,a}.
                    }
                }
            }
            
            // --- Step 11: Update Threshold [cite: 147, 169] ---
            // "Added to a larger one by the layered stepping value"
            L_value_threshold += alpha;

            // --- Tentative Decision & Syndrome Check [cite: 94, 97] ---
            bool valid_codeword = true;
            
            // 1. Compute Posterior LLR and Decision
            for (int n = 0; n < N; ++n) {
                double L_posterior = received_LLR[n];
                for (int m : H_cols[n]) {
                    L_posterior += R[m][n];
                }
                decoded_bits[n] = (L_posterior < 0) ? 1 : 0;
            }
            
            // 2. Check Syndrome: c * H^T = 0
            for (int m = 0; m < M; ++m) {
                int parity = 0;
                for (int n : H_rows[m]) {
                    parity ^= decoded_bits[n];
                }
                if (parity != 0) {
                    valid_codeword = false;
                    break;
                }
            }
            
            if (valid_codeword) {
                std::cout << "Converged at iteration " << l << std::endl;
                return decoded_bits;
            }
        }
        
        std::cout << "Reached max iterations without convergence." << std::endl;
        return decoded_bits;
    }
};

// // --- Example Usage ---
// int main() {
//     // Example: Simple (6,3) code for demonstration
//     // H matrix:
//     // 1 1 0 1 0 0
//     // 0 1 1 0 1 0
//     // 1 0 1 0 0 1
//     std::vector<std::vector<int>> H = {
//         {1, 1, 0, 1, 0, 0},
//         {0, 1, 1, 0, 1, 0},
//         {1, 0, 1, 0, 0, 1}
//     };

//     // Received LLRs (Example: Noisy transmission of all-zeros codeword)
//     // Positive LLR -> likely 0, Negative LLR -> likely 1
//     // Index 1 (value 0.2) is the least reliable.
//     std::vector<double> received_LLR = {2.5, 0.2, 3.1, 2.8, 1.5, 4.0};

//     // Parameters
//     int max_iter = 50;
//     double alpha = 0.5; // Layered stepping value [cite: 133]

//     RBLBP_Decoder decoder(3, 6, H);
//     std::vector<int> result = decoder.decode(received_LLR, max_iter, alpha);

//     std::cout << "Decoded Codeword: ";
//     for (int bit : result) std::cout << bit << " ";
//     std::cout << std::endl;

//     return 0;
// }