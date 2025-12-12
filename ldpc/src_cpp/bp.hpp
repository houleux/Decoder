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

                            double tanh_arg = std::tanh(temp / 2.0);
                            if (std::abs(tanh_arg) < EPS_TANH) {
                                tanh_arg = (tanh_arg >= 0.0 ? EPS_TANH : -EPS_TANH);
                            }

                            double psi = std::log(std::abs(tanh_arg));

                            int sign_Lmj = (edge.bit_to_check_msg < 0.0) ? -1 : 1;
                            int sign_factor = sm * sign_Lmj;

                            double newR = - (double)sign_factor * psi;

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

            // Node-wise Residual Belief Propagation (node-wise RBP / ARBP)
            // - max_updates: stop after this many check-node selections (<=0 = no limit)
            // - residual_eps: stop if top residual <= residual_eps (<=0 disables)
            // - use_approx_residual: if true compute residuals with min-sum approx (ARBP), else exact sum-product residual
            // - check_syndrome_cb: optional callback to stop early if decoded word valid; pass nullptr to ignore
            void nodewise_rbp(int max_updates = -1,
                            double residual_eps = 0.0,
                            bool use_approx_residual = true,
                            std::function<bool()> check_syndrome_cb = nullptr)
            {
                const double EPS_TANH = 1e-12;

                // helper: unique key for an edge (m, j)
                auto edge_key = [&](int m, int j) -> int {
                    return m * this->bit_count + j;
                };

                // Sum-product exact check->var computation for propagation (returns new R_mj)
                auto compute_check_to_var_sumproduct = [&](int m, int exclude_col) -> double {
                    double Am = 0.0;
                    int sm = 1;
                    // compute Am and product of signs
                    for (auto &edge : pcm.iterate_row(m)) {
                        double Lmn = edge.bit_to_check_msg;
                        double t = std::tanh(Lmn / 2.0);
                        if (std::abs(t) < EPS_TANH) t = std::copysign(EPS_TANH, t);
                        Am += std::log(std::abs(t));
                        if (Lmn < 0.0) sm = -sm;
                    }
                    // find L_mj for excluded edge
                    double Lmj = 0.0;
                    for (auto &edge : pcm.iterate_row(m)) {
                        if (edge.col_index == exclude_col) { Lmj = edge.bit_to_check_msg; break; }
                    }
                    double t_self = std::tanh(Lmj / 2.0);
                    if (std::abs(t_self) < EPS_TANH) t_self = std::copysign(EPS_TANH, t_self);
                    double log_abs_t_self = std::log(std::abs(t_self));
                    double temp = Am - log_abs_t_self;

                    double tanh_arg = std::tanh(temp / 2.0);
                    if (std::abs(tanh_arg) < EPS_TANH) tanh_arg = std::copysign(EPS_TANH, tanh_arg);
                    double psi = std::log(std::abs(tanh_arg)); // <= 0

                    int sign_Lmj = (Lmj < 0.0) ? -1 : 1;
                    int sign_factor = sm * sign_Lmj;
                    double newR = - static_cast<double>(sign_factor) * psi;
                    return newR;
                };

                // Min-sum approximation for residual computation only (ARBP)
                auto compute_check_to_var_minsum_approx = [&](int m, int exclude_col) -> double {
                    double min1 = std::numeric_limits<double>::infinity();
                    double min2 = std::numeric_limits<double>::infinity();
                    int sign_product = 1;
                    int sign_excluded = 1;

                    for (auto &edge : pcm.iterate_row(m)) {
                        if (edge.col_index == exclude_col) {
                            sign_excluded = (edge.bit_to_check_msg < 0.0) ? -1 : 1;
                            continue;
                        }
                        double Labs = std::abs(edge.bit_to_check_msg);
                        if (Labs <= min1) { min2 = min1; min1 = Labs; }
                        else if (Labs < min2) { min2 = Labs; }
                        if (edge.bit_to_check_msg < 0.0) sign_product = -sign_product;
                    }

                    int sign_factor = sign_product * sign_excluded;
                    double approx_mag = (min1 == std::numeric_limits<double>::infinity()) ? 0.0 : min1;
                    double t = std::tanh(approx_mag / 2.0);
                    if (std::abs(t) < EPS_TANH) t = std::copysign(EPS_TANH, t);
                    double psi = std::log(std::abs(t)); // <=0
                    double newR_approx = - static_cast<double>(sign_factor) * psi;
                    return newR_approx;
                };

                // 1) initialize bit_to_check_msg for all edges from current L(q_j) and stored R_mj
                for (int col = 0; col < this->bit_count; ++col) {
                    for (auto &edge : pcm.iterate_column(col)) {
                        edge.bit_to_check_msg = this->log_prob_ratios[col] - edge.check_to_bit_msg;
                    }
                }

                // 2) prepare priority queue of residuals (max-heap)
                struct ResidualEntry { double residual; int m; int col; };
                struct ResidualCmp { bool operator()(ResidualEntry const &a, ResidualEntry const &b) const { return a.residual < b.residual; } };
                std::priority_queue<ResidualEntry, std::vector<ResidualEntry>, ResidualCmp> pq;

                std::unordered_map<int,double> current_residual; // edge_key -> residual
                current_residual.reserve(this->check_count * 2);

                // compute initial residuals
                for (int m = 0; m < this->check_count; ++m) {
                    for (auto &edge : pcm.iterate_row(m)) {
                        double newR = use_approx_residual ? compute_check_to_var_minsum_approx(m, edge.col_index)
                                                        : compute_check_to_var_sumproduct(m, edge.col_index);
                        double oldR = edge.check_to_bit_msg;
                        double r = std::abs(newR - oldR);
                        int k = edge_key(m, edge.col_index);
                        current_residual[k] = r;
                        pq.push({r, m, edge.col_index});
                    }
                }

                // main loop
                int updates_done = 0;
                while (!pq.empty()) {
                    // pop top, skip stale
                    ResidualEntry top = pq.top(); pq.pop();
                    int k = edge_key(top.m, top.col);
                    auto it = current_residual.find(k);
                    if (it == current_residual.end()) continue;
                    if (std::abs(it->second - top.residual) > 1e-15) continue; // stale

                    if (residual_eps > 0.0 && top.residual <= residual_eps) break;

                    int chosen_m = top.m;

                    // 3) For chosen check node: compute exact new R for all its outgoing edges (propagate full sum-product)
                    std::vector<std::pair<int,double>> updates; updates.reserve(32);
                    for (auto &edge : pcm.iterate_row(chosen_m)) {
                        double newR_exact = compute_check_to_var_sumproduct(chosen_m, edge.col_index);
                        updates.emplace_back(edge.col_index, newR_exact);
                    }

                    // 4) apply updates: store oldR, write newR to edges, update global L(q_j) incrementally
                    for (auto &p : updates) {
                        int col = p.first;
                        double newR = p.second;

                        // find reference to the edge to read oldR and update it
                        double oldR = 0.0;
                        for (auto &edge : pcm.iterate_row(chosen_m)) {
                            if (edge.col_index == col) { oldR = edge.check_to_bit_msg; break; }
                        }
                        // now write newR into that edge
                        for (auto &edge : pcm.iterate_row(chosen_m)) {
                            if (edge.col_index == col) { edge.check_to_bit_msg = newR; break; }
                        }

                        // incremental update to L(q_j)
                        this->log_prob_ratios[col] += (newR - oldR);
                    }

                    // 5) update bit_to_check_msg for affected variable nodes and recompute residuals
                    std::vector<int> affected_checks;
                    affected_checks.reserve(64);
                    for (auto &edge : pcm.iterate_row(chosen_m)) {
                        int col = edge.col_index;
                        for (auto &e2 : pcm.iterate_column(col)) {
                            // update bit_to_check_msg now that R may have changed
                            e2.bit_to_check_msg = this->log_prob_ratios[col] - e2.check_to_bit_msg;
                            affected_checks.push_back(e2.row_index);
                        }
                    }

                    // remove duplicates
                    std::sort(affected_checks.begin(), affected_checks.end());
                    affected_checks.erase(std::unique(affected_checks.begin(), affected_checks.end()), affected_checks.end());

                    // recompute residuals for edges in affected checks and push to pq
                    for (int m : affected_checks) {
                        for (auto &edge : pcm.iterate_row(m)) {
                            double approxNewR = use_approx_residual ? compute_check_to_var_minsum_approx(m, edge.col_index)
                                                                    : compute_check_to_var_sumproduct(m, edge.col_index);
                            double oldR = edge.check_to_bit_msg;
                            double r = std::abs(approxNewR - oldR);
                            int key = edge_key(m, edge.col_index);
                            current_residual[key] = r;
                            pq.push({r, m, edge.col_index});
                        }
                    }

                    ++updates_done;
                    if (max_updates > 0 && updates_done >= max_updates) break;

                    if (check_syndrome_cb) {
                        if (check_syndrome_cb()) break;
                    }
                } // end while
            }


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
