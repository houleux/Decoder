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

            // NOTE: Kept this arround just in case 
    
            // void initialise_log_domain_bp() {
            //     for (int i = 0; i < this->bit_count; i++) {
            //         this->initial_log_prob_ratios[i] = std::log(
            //                 (1 - this->channel_probabilities[i]) / this->channel_probabilities[i]);

            //         for (auto &e: this->pcm.iterate_column(i)) {
            //             e.bit_to_check_msg = this->initial_log_prob_ratios[i];
            //         }
            //     }
            // }

            // NOTE: initialise log_domain only does i_llr = llr; bit_check_msg = i_llr
            void initialise_log_domain_bp(const std::vector<double> &llr_vector) {
                for (int i = 0; i < this->bit_count; i++) {
                    this->initial_log_prob_ratios[i] = llr_vector[i];

                    for (auto &e: this->pcm.iterate_column(i)) {
                        e.bit_to_check_msg = this->initial_log_prob_ratios[i];
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

            // TODO: Check if the function is algorithmically correct or not

            void bp_decode_cluster(std::vector<double> &llr_vector, const std::vector<int> &cluster_checks) {
                if (llr_vector.size() != static_cast<size_t>(bit_count)) {
                    throw std::runtime_error("LLR vector length does not match number of variable nodes");
                }
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

                // initialise_log_domain_bp(llr_vector);

                // Bit-to-check messages

                for (int i = 0; i < bit_count; i++) {
                    for (auto &e: this->pcm.iterate_column(i)) {
                        e.bit_to_check_msg = llr_vector[i];
                    }
                }


                // Only the selected checks respond and compute check-to-bit messages
                if (bp_method == PRODUCT_SUM) {
                    for (int check_index: cluster_checks) {
                        double prefix = 1.0;
                        for (auto &edge: pcm.iterate_row(check_index)) {
                            edge.check_to_bit_msg = prefix;
                            prefix *= std::tanh(edge.bit_to_check_msg / 2.0);
                        }

                        double suffix = 1.0;
                        for (auto &edge: pcm.reverse_iterate_row(check_index)) {
                            edge.check_to_bit_msg *= suffix;
                            edge.check_to_bit_msg =
                                    std::log((1.0 + edge.check_to_bit_msg) /
                                             (1.0 - edge.check_to_bit_msg));
                            suffix *= std::tanh(edge.bit_to_check_msg / 2.0);
                        }
                    }
                } else { // MINIMUM_SUM
                    throw std::runtime_error("Cluster decoding with Minimum-Sum method is not yet implemented");
                }

                // Accumulate only the participating check contributions back into the LLR vector
                for (int col = 0; col < bit_count; ++col) {
                    double updated = llr_vector[col];
                    for (auto &edge: pcm.iterate_column(col)) {
                        if (check_mask[edge.row_index]) {
                            updated += edge.check_to_bit_msg;
                        }
                    }
                    llr_vector[col] = updated;
                }
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
