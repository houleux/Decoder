#ifndef BP_H
#define BP_H

#include <utility>
#include <vector>
#include <memory>
#include <iterator>
#include <cmath>
#include <limits>
#include <random>
#include <chrono>
#include <stdexcept> // required for std::runtime_error
#include <set>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <algorithm>


#include "math.h"
#include "sparse_matrix_base.hpp"
#include "gf2sparse.hpp"
#include "rng.hpp"

namespace ldpc {
    namespace bp {

        enum BpMethod {
            PRODUCT_SUM = 0,
            MINIMUM_SUM = 1
        };

        enum BpSchedule {
            SERIAL = 0,
            PARALLEL = 1,
            SERIAL_RELATIVE = 2,
            CLUSTER = 3
        };

        enum BpInputType {
            SYNDROME = 0,
            RECEIVED_VECTOR = 1,
            AUTO = 2
        };

        const std::vector<int> NULL_INT_VECTOR = {};

        class BpEntry : public ldpc::sparse_matrix_base::EntryBase<BpEntry> {
        public:
            double bit_to_check_msg = 0.0;
            double check_to_bit_msg = 0.0;

            ~BpEntry() = default;
        };
        using BpSparse = ldpc::gf2sparse::GF2Sparse<BpEntry>;

        // FIX: Remove the useless/unused props/variables from this class

        class BpDecoder {
        public:
            BpSparse &pcm;
            int check_count;
            int bit_count;
            int maximum_iterations;
            BpMethod bp_method;
            BpSchedule schedule;
            double ms_scaling_factor;
            std::vector<uint8_t> decoding;
            std::vector<uint8_t> candidate_syndrome;

            std::vector<double> log_prob_ratios;
            std::vector<double> initial_log_prob_ratios;
            int iterations;
            bool converge;

            BpDecoder(
                    BpSparse &parity_check_matrix,
                    int maximum_iterations = 0,
                    BpMethod bp_method = PRODUCT_SUM,
                    BpSchedule schedule = PARALLEL,
                    double min_sum_scaling_factor = 0.625) :
                    pcm(parity_check_matrix),
                    check_count(pcm.m), bit_count(pcm.n), maximum_iterations(maximum_iterations), bp_method(bp_method),
                    schedule(schedule), ms_scaling_factor(min_sum_scaling_factor),
                    iterations(0) //the parity check matrix is passed in by reference
            {

                this->initial_log_prob_ratios.resize(bit_count);
                this->log_prob_ratios.resize(bit_count);
                this->candidate_syndrome.resize(check_count);
                this->decoding.resize(bit_count);
                this->converge = 0;
            }

            ~BpDecoder() = default;

            void reset() {
                this->iterations = 0;
                this->converge = false;
                
                std::fill(this->decoding.begin(), this->decoding.end(), 0);
                std::fill(this->candidate_syndrome.begin(), this->candidate_syndrome.end(), 0);
                std::fill(this->log_prob_ratios.begin(), this->log_prob_ratios.end(), 0.0);
                std::fill(this->initial_log_prob_ratios.begin(), this->initial_log_prob_ratios.end(), 0.0);

                for (int i = 0; i < this->bit_count; i++) {
                    for (auto &e: this->pcm.iterate_column(i)) {
                        e.bit_to_check_msg = 0.0;
                        e.check_to_bit_msg = 0.0;
                    }
                }
            }

            // NOTE: initialise log_domain only does i_llr = llr; bit_check_msg = i_llr
            void initialise_log_domain_bp(const std::vector<double> &llr_vector_channel) {
                for (int i = 0; i < this->bit_count; i++) {
                    this->initial_log_prob_ratios[i] = llr_vector_channel[i];
                    this->log_prob_ratios[i] = llr_vector_channel[i];

                    for (auto &e: this->pcm.iterate_column(i)) {
                        e.bit_to_check_msg = llr_vector_channel[i];
                        e.check_to_bit_msg = 0.0;
                    }
                }
            }

            std::vector<uint8_t> decode(std::vector<double> &llr_vector) {
                if (llr_vector.size() != this->bit_count) {
                    throw std::runtime_error("Input vector length does not match bit count");
                }

                if (schedule == PARALLEL) {
                    return bp_decode_parallel(llr_vector);
                }
                if (schedule == SERIAL || schedule == SERIAL_RELATIVE) {
                    throw std::runtime_error("Serial schedules are not yet implemented for AWGN decoding");
                }
                if (schedule == CLUSTER) {
                    throw std::runtime_error("Cluster schedule must be invoked via bp_decode_cluster");
                }
                throw std::runtime_error("Invalid BP schedule");
            }


            void bp_decode_cluster(const std::vector<int> &cluster_checks) {
                if (cluster_checks.empty()) {
                    return; // nothing to update
                }

                std::vector<uint8_t> check_mask(check_count, 0);
                for (int check_index: cluster_checks) {
                    if (check_index < 0 || check_index >= check_count) {
                        throw std::runtime_error("Cluster contains invalid check index");
                    }
                    check_mask[check_index] = 1;
                }

                const double EPS_TANH = 1e-12;
                const double MIN_ARG = 1e-308;


                for (int col = 0; col < this->bit_count; ++col) {
                    for (auto &edge : pcm.iterate_column(col)) {
                        if (check_mask[edge.row_index]) {
                            edge.bit_to_check_msg = this->log_prob_ratios[col] - edge.check_to_bit_msg; 
                        }
                    }
                }

                if (bp_method == PRODUCT_SUM) {
                    for (int check_index : cluster_checks) {
                        double Am = 0.0;
                        for (auto &edge : pcm.iterate_row(check_index)) {
                            double t = std::tanh(edge.bit_to_check_msg / 2.0);
                            if (std::abs(t) < EPS_TANH) {
                                t = (t >= 0) ? EPS_TANH : -EPS_TANH;
                            }
                            Am += std::log(std::abs(t));
                        }

                        int sm = 1;
                        for (auto &edge : pcm.iterate_row(check_index)) {
                            if (edge.bit_to_check_msg < 0.0) sm = -sm;
                        }

                        for (auto &edge : pcm.iterate_row(check_index)) {
                            double oldR = edge.check_to_bit_msg;

                            double t_self = std::tanh(edge.bit_to_check_msg/2.0);
                            if (std::abs(t_self) < EPS_TANH) {
                                t_self = (t_self >= 0.0 ? EPS_TANH : -EPS_TANH);
                            }

                            double log_abs_t_self = std::log(std::abs(t_self));

                            double temp = Am - log_abs_t_self;

                            int sign_Lmj = (edge.bit_to_check_msg < 0.0) ? -1 : 1;
                            int sign_factor = sm * sign_Lmj;

                            double prod_others = sign_factor * std::exp(temp);

                            // Clamp prod_others to avoid singularity at +/- 1
                            if (prod_others > 1.0 - 1e-15) prod_others = 1.0 - 1e-15;
                            if (prod_others < -1.0 + 1e-15) prod_others = -1.0 + 1e-15;

                            double newR = std::log((1.0 + prod_others) / (1.0 - prod_others));

                            if (!std::isfinite(newR)) {
                                // fallback — small value or clamp
                                if (std::isnan(newR)) newR = 0.0;
                                else if (newR > 1e300) newR = 1e300;
                                else if (newR < -1e300) newR = -1e300;
                            }

                            edge.check_to_bit_msg = newR;

                            this->log_prob_ratios[edge.col_index] += (newR - oldR);

                        }
                    }
                    
                } else { // MINIMUM_SUM
                    throw std::runtime_error("Cluster decoding with Minimum-Sum method is not yet implemented");
                }

            }

            std::vector<double> get_residuals() {
                std::vector<double> residuals(this->check_count, 0.0);

                if (this->bp_method == PRODUCT_SUM) {
                    for (int row = 0; row < this->check_count; ++row) {
                        double max_residual = 0.0;
                        
                        const double EPS_TANH = 1e-12;
                        double Am = 0.0;
                        int sm = 1;
                        for (auto &edge : pcm.iterate_row(row)) {
                            double t = std::tanh(edge.bit_to_check_msg / 2.0);
                            if (std::abs(t) < EPS_TANH) {
                                t = (t >= 0) ? EPS_TANH : -EPS_TANH;
                            }
                            Am += std::log(std::abs(t));
                            if (edge.bit_to_check_msg < 0.0) sm = -sm;
                        }

                        for (auto &edge : pcm.iterate_row(row)) {
                            double old_msg = edge.check_to_bit_msg;
                            
                            double t_self = std::tanh(edge.bit_to_check_msg / 2.0);
                            if (std::abs(t_self) < EPS_TANH) {
                                t_self = (t_self >= 0.0 ? EPS_TANH : -EPS_TANH);
                            }
                            double log_abs_t_self = std::log(std::abs(t_self));
                            
                            double temp = Am - log_abs_t_self; // log(|prod_others|)
                            
                            int sign_Lmj = (edge.bit_to_check_msg < 0.0) ? -1 : 1;
                            int sign_factor = sm * sign_Lmj;
                            
                            double prod_others = sign_factor * std::exp(temp);
                            
                            // Clamp prod_others to avoid singularity at +/- 1
                            if (prod_others > 1.0 - 1e-15) prod_others = 1.0 - 1e-15;
                            if (prod_others < -1.0 + 1e-15) prod_others = -1.0 + 1e-15;
                            
                            double new_msg = std::log((1.0 + prod_others) / (1.0 - prod_others));
                            
                            double residual = std::abs(new_msg - old_msg);
                            if (residual > max_residual) max_residual = residual;
                        }
                        residuals[row] = max_residual;
                    }
                } else if (this->bp_method == MINIMUM_SUM) {
                    throw std::runtime_error("Minsum is not implemented yet");
                }

                return residuals;
            }

            // Approximate mutual-information residuals by reusing raw residual magnitudes.
            // This keeps the Python binding functional without a bespoke MI transform.
            std::vector<double> get_mi_residuals() {
                return get_residuals();
            }

            double J_func(const double sigma) {
                double mi = 0.0;

                if (sigma >= 10) {
                    mi = 1.0;
                }
                else if (sigma > 1.6363) {
                    mi = 1.0 - std::exp(0.001815 * sigma * sigma * sigma - 0.142675 * sigma * sigma - 0.082205 * sigma + 0.054960);
                }
                else {
                    mi = -0.0421061 * sigma * sigma * sigma + 0.209252 * sigma * sigma - 0.00640081 * sigma;
                }

                return mi;
            }

            double J_inv_func(const double mi) {
                double sigma = 0.0;

                if (mi <= 0.3646) {
                    sigma = 1.09542 * mi * mi + 0.214217 * mi + 2.33727 * std::sqrt(mi);
                }

                else {
                    sigma = -0.706692 * std::log(0.386013 * (1.0 - mi)) - 1.75017 * mi;
                }

                return sigma;
            }

            std::vector<int> m2i2_scheduler(const std::vector<std::vector<int>> &P, double code_rate, double EbN0) {
                // Schedule to be returned
                std::vector<int> schedule;
                
                // Get dimensions from base matrix P
                int Mp = P.size();
                if (Mp == 0) {
                    throw std::runtime_error("Base matrix P is empty");
                }
                int Np = P[0].size();
                
                // Initialize data structures
                // u[i][j]: update counter for edge (i,j)
                std::vector<std::vector<int>> u(Mp, std::vector<int>(Np, 0));
                
                // I_EC[i][j]: check-to-variable MI
                std::vector<std::vector<double>> I_EC(Mp, std::vector<double>(Np, 0.0));
                
                // I_EV[i][j]: variable-to-check MI
                std::vector<std::vector<double>> I_EV(Mp, std::vector<double>(Np, 0.0));
                
                // I_ch[j]: channel MI
                std::vector<double> I_ch(Np, 0.0);
                
                // Predicted check-to-variable MI
                std::vector<std::vector<double>> Ip_EC(Mp, std::vector<double>(Np, 0.0));
                
                // Cluster-wise MI increase
                std::vector<double> R_cluster(Mp, 0.0);
                
                // Combined MI per variable node
                std::vector<double> I_CMI(Np, 0.0);
                
                // Initialize channel MI for all variable nodes
                double sigma_ch = std::sqrt(8.0 * code_rate * EbN0);
                for (int j = 0; j < Np; ++j) {
                    I_ch[j] = J_func(sigma_ch);
                }
                
                // Initialize I_EV with channel MI for existing edges (where P[i][j] != -1)
                for (int i = 0; i < Mp; ++i) {
                    for (int j = 0; j < Np; ++j) {
                        if (P[i][j] != -1) {
                            I_EV[i][j] = I_ch[j];
                        }
                    }
                }
                
                // Main scheduling loop
                while (true) {
                    // Step 1: Predict MI for all edges
                    for (int i = 0; i < Mp; ++i) {
                        for (int j = 0; j < Np; ++j) {
                            if (P[i][j] != -1) {
                                // Compute sum of squared J_inv over neighbors excluding current edge
                                double sum_sq = 0.0;
                                for (int b = 0; b < Np; ++b) {
                                    if (b != j && P[i][b] != -1) {
                                        double ji = J_inv_func(1.0 - I_EV[i][b]);
                                        sum_sq += ji * ji;
                                    }
                                }
                                
                                Ip_EC[i][j] = 1.0 - J_func(std::sqrt(sum_sq));
                            }
                        }
                    }
                    
                    // Step 2: Compute cluster-wise MI increase
                    for (int i = 0; i < Mp; ++i) {
                        R_cluster[i] = 0.0;
                        for (int j = 0; j < Np; ++j) {
                            if (P[i][j] != -1) {
                                R_cluster[i] += Ip_EC[i][j] - I_EC[i][j];
                            }
                        }
                    }
                    
                    // Step 3: Select best cluster
                    int i_star = 0;
                    double max_increase = R_cluster[0];
                    for (int i = 1; i < Mp; ++i) {
                        if (R_cluster[i] > max_increase) {
                            max_increase = R_cluster[i];
                            i_star = i;
                        }
                    }
                    
                    // Append selected cluster to schedule
                    schedule.push_back(i_star);
                    
                    // Step 4: Commit updates for selected cluster
                    for (int j = 0; j < Np; ++j) {
                        if (P[i_star][j] != -1) {
                            I_EC[i_star][j] = Ip_EC[i_star][j];
                            u[i_star][j] += 1;
                        }
                    }
                    
                    // Step 5: Update affected V2C MI
                    // For each variable node j connected to the selected check node i_star
                    for (int j = 0; j < Np; ++j) {
                        if (P[i_star][j] != -1) {
                            // For all check nodes 'a' connected to variable node j
                            for (int a = 0; a < Mp; ++a) {
                                if (P[a][j] != -1) {
                                    // Compute updated V2C MI for edge (a,j)
                                    double sum_sq = 0.0;
                                    
                                    // Sum over all check nodes connected to vj except 'a'
                                    for (int c = 0; c < Mp; ++c) {
                                        if (c != a && P[c][j] != -1) {
                                            double ji = J_inv_func(I_EC[c][j]);
                                            sum_sq += ji * ji;
                                        }
                                    }
                                    
                                    // Add channel contribution
                                    double ji_ch = J_inv_func(I_ch[j]);
                                    sum_sq += ji_ch * ji_ch;
                                    
                                    I_EV[a][j] = J_func(std::sqrt(sum_sq));
                                }
                            }
                        }
                    }
                    
                    // Step 6: Check termination
                    bool converged = true;
                    for (int j = 0; j < Np; ++j) {
                        double sum_sq = 0.0;
                        
                        // Sum over all check nodes connected to variable node j
                        for (int a = 0; a < Mp; ++a) {
                            if (P[a][j] != -1) {
                                double ji = J_inv_func(I_EC[a][j]);
                                sum_sq += ji * ji;
                            }
                        }
                        
                        // Add channel contribution
                        double ji_ch = J_inv_func(I_ch[j]);
                        sum_sq += ji_ch * ji_ch;
                        
                        I_CMI[j] = J_func(std::sqrt(sum_sq));
                        
                        if (I_CMI[j] < 1.0) {
                            converged = false;
                        }
                    }
                    
                    if (converged) {
                        break;
                    }
                }
                
                return schedule;
            }

            

            // std::vector<double> get_mi_residuals() {
            //     std::vector<double> residuals(this->check_count, 0.0);

            //     if (this->bp_method == PRODUCT_SUM) {
            //         for (int row = 0; row < this->check_count; ++row) {
            //             double max_residual = 0.0;
                        
            //             const double EPS_TANH = 1e-12;
            //             double Am = 0.0;
            //             int sm = 1;
            //             for (auto &edge : pcm.iterate_row(row)) {
            //                 double t = std::tanh(edge.bit_to_check_msg / 2.0);
            //                 if (std::abs(t) < EPS_TANH) {
            //                     t = (t >= 0) ? EPS_TANH : -EPS_TANH;
            //                 }
            //                 Am += std::log(std::abs(t));
            //                 if (edge.bit_to_check_msg < 0.0) sm = -sm;
            //             }

            //             for (auto &edge : pcm.iterate_row(row)) {
            //                 double old_msg_mi = LLR_to_MI(std::vector<double>{edge.check_to_bit_msg});
                            
            //                 double t_self = std::tanh(edge.bit_to_check_msg / 2.0);
            //                 if (std::abs(t_self) < EPS_TANH) {
            //                     t_self = (t_self >= 0.0 ? EPS_TANH : -EPS_TANH);
            //                 }
            //                 double log_abs_t_self = std::log(std::abs(t_self));
                            
            //                 double temp = Am - log_abs_t_self; // log(|prod_others|)
                            
            //                 int sign_Lmj = (edge.bit_to_check_msg < 0.0) ? -1 : 1;
            //                 int sign_factor = sm * sign_Lmj;
                            
            //                 double prod_others = sign_factor * std::exp(temp);
                            
            //                 // Clamp prod_others to avoid singularity at +/- 1
            //                 if (prod_others > 1.0 - 1e-15) prod_others = 1.0 - 1e-15;
            //                 if (prod_others < -1.0 + 1e-15) prod_others = -1.0 + 1e-15;
                            
            //                 double new_msg = std::log((1.0 + prod_others) / (1.0 - prod_others));
            //                 double new_msg_mi = LLR_to_MI(std::vector<double>{new_msg});
                            
            //                 double residual = std::abs(new_msg_mi - old_msg_mi);
            //                 if (residual > max_residual) max_residual = residual;
            //             }
            //             residuals[row] = max_residual;
            //         }
            //     } else if (this->bp_method == MINIMUM_SUM) {
            //          throw std::runtime_error("MI residuals for Minimum-Sum method are not yet implemented");
            //     }

            //     return residuals;
            // }
            // TODO: Check if function is correct/ and compare against matlab flood decoder

            std::vector<uint8_t> &bp_decode_parallel(const std::vector<double> &llr_vector) {
                if (llr_vector.size() != static_cast<size_t>(bit_count)) {
                    throw std::runtime_error("LLR vector length does not match number of variable nodes");
                }

                converge = false;
                iterations = 0;
                initialise_log_domain_bp(llr_vector);

                for (int it = 1; it <= maximum_iterations; ++it) {
                    // Check-to-bit updates
                    if (bp_method == PRODUCT_SUM) {
                        for (int row = 0; row < check_count; ++row) {
                            double prefix = 1.0;
                            for (auto &edge : pcm.iterate_row(row)) {
                                edge.check_to_bit_msg = prefix;
                                prefix *= std::tanh(edge.bit_to_check_msg / 2.0);
                            }

                            double suffix = 1.0;
                            for (auto &edge : pcm.reverse_iterate_row(row)) {
                                edge.check_to_bit_msg *= suffix;
                                edge.check_to_bit_msg =
                                        std::log((1.0 + edge.check_to_bit_msg) /
                                                (1.0 - edge.check_to_bit_msg));
                                suffix *= std::tanh(edge.bit_to_check_msg / 2.0);
                            }
                        }
                    } else { // MINIMUM_SUM
                        for (int row = 0; row < check_count; ++row) {
                            int total_sgn = 0;
                            double min_val = std::numeric_limits<double>::max();

                            for (auto &edge : pcm.iterate_row(row)) {
                                if (edge.bit_to_check_msg <= 0.0) {
                                    total_sgn ^= 1;
                                }
                                edge.check_to_bit_msg = min_val;
                                const double abs_val = std::abs(edge.bit_to_check_msg);
                                if (abs_val < min_val) {
                                    min_val = abs_val;
                                }
                            }

                            min_val = std::numeric_limits<double>::max();
                            for (auto &edge : pcm.reverse_iterate_row(row)) {
                                int sgn = total_sgn;
                                if (edge.bit_to_check_msg <= 0.0) {
                                    sgn ^= 1;
                                }
                                if (min_val < edge.check_to_bit_msg) {
                                    edge.check_to_bit_msg = min_val;
                                }
                                const double message_sign = (sgn % 2 == 0) ? 1.0 : -1.0;
                                edge.check_to_bit_msg *= message_sign * ms_scaling_factor;

                                const double abs_val = std::abs(edge.bit_to_check_msg);
                                if (abs_val < min_val) {
                                    min_val = abs_val;
                                }
                            }
                        }
                    }

                    // Bit updates + hard decisions
                    std::fill(candidate_syndrome.begin(), candidate_syndrome.end(), 0);
                    for (int col = 0; col < bit_count; ++col) {
                        double temp = initial_log_prob_ratios[col];
                        for (auto &edge : pcm.iterate_column(col)) {
                            edge.bit_to_check_msg = temp;
                            temp += edge.check_to_bit_msg;
                        }

                        log_prob_ratios[col] = temp;
                        decoding[col] = (temp <= 0.0);
                        if (decoding[col]) {
                            for (auto &edge : pcm.iterate_column(col)) {
                                candidate_syndrome[edge.row_index] ^= 1;
                            }
                        }
                    }

                    if (std::all_of(candidate_syndrome.begin(), candidate_syndrome.end(),
                                    [](uint8_t s) { return s == 0; })) {
                        converge = true;
                        iterations = it;
                        break;
                    }
                    iterations = it;

                    // Bit-to-check updates
                    for (int col = 0; col < bit_count; ++col) {
                        double suffix = 0.0;
                        for (auto &edge : pcm.reverse_iterate_column(col)) {
                            edge.bit_to_check_msg += suffix;
                            suffix += edge.check_to_bit_msg;
                        }
                    }
                }

                return decoding;
            }

            // TODO: Evaluate if these functions are needed or not
            std::vector<uint8_t> &bp_decode_single_scan(std::vector<uint8_t> &syndrome) {
                throw std::runtime_error("Single-scan decoding is not implemented for AWGN decoding yet");
            }

            std::vector<uint8_t> &bp_decode_serial(std::vector<double> &llr_vector) {
                throw std::runtime_error("Serial schedules are not implemented for AWGN decoding yet");
            }

            std::vector<uint8_t> &
            soft_info_decode_serial(std::vector<double> &soft_info_syndrome, double cutoff, double sigma) {
                throw std::runtime_error("soft info decoding is not implemented for AWGN decoding yet");
            }
        };
    }
}  // namespace ldpc::bp

#endif
