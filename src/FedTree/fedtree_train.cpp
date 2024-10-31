//
// Created by liqinbin on 10/13/20.
//

#include "FedTree/FL/FLparam.h"
#include "FedTree/FL/FLtrainer.h"
#include "FedTree/FL/partition.h"
#include "FedTree/parser.h"
#include "FedTree/dataset.h"
#include "FedTree/Tree/gbdt.h"
#include <algorithm>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <cmath>
#include <fstream>

#ifdef _WIN32
INITIALIZE_EASYLOGGINGPP
#endif

std::vector<std::vector<int>> power_set(const std::vector<int>& set) {
    std::vector<std::vector<int>> subsets;
    int n = set.size();
    int subset_count = 1 << n;  // 2^n 个子集

    for (int mask = 0; mask < subset_count; mask++) {
        std::vector<int> subset;
        for (int i = 0; i < n; i++) {
            if (mask & (1 << i)) {
                subset.push_back(set[i]);
            }
        }
        subsets.push_back(subset);
    }
    return subsets;
}

int main(int argc, char** argv){
    std::chrono::high_resolution_clock timer;
    el::Loggers::reconfigureAllLoggers(el::ConfigurationType::Format, "%datetime %level %fbase:%line : %msg");
    el::Loggers::addFlag(el::LoggingFlag::ColoredTerminalOutput);
    el::Loggers::addFlag(el::LoggingFlag::FixedTimeFormat);

    // centralized training test
    FLParam fl_param;
    Parser parser;
    if (argc > 1) {
        // flag
        parser.parse_param(fl_param, argv[1]);
    } else {
        printf("Usage: <config file path> \n");
        exit(0);
    }
    GBDTParam &model_param = fl_param.gbdt_param;
    if(model_param.verbose == 0) {
        el::Loggers::reconfigureAllLoggers(el::Level::Debug, el::ConfigurationType::Enabled, "false");
        el::Loggers::reconfigureAllLoggers(el::Level::Trace, el::ConfigurationType::Enabled, "false");
        el::Loggers::reconfigureAllLoggers(el::Level::Info, el::ConfigurationType::Enabled, "false");
    }
    else if (model_param.verbose == 1) {
        el::Loggers::reconfigureAllLoggers(el::Level::Debug, el::ConfigurationType::Enabled, "false");
        el::Loggers::reconfigureAllLoggers(el::Level::Trace, el::ConfigurationType::Enabled, "false");
    }

    if (!model_param.profiling) {
        el::Loggers::reconfigureAllLoggers(el::ConfigurationType::PerformanceTracking, "false");
    }
    int n_parties = fl_param.n_parties;

    if (!fl_param.partition || model_param.paths.size() > 1) {
        CHECK_EQ(n_parties, model_param.paths.size());
        fl_param.partition = false;
    }

    // 数据集处理
    vector<DataSet> train_subsets(n_parties);
    vector<DataSet> test_subsets(n_parties);
    vector<DataSet> subsets(n_parties);
    vector<SyncArray<bool>> feature_map(n_parties);
    std::map<int, vector<int>> batch_idxs;
    DataSet dataset;
    bool use_global_test_set = !model_param.test_path.empty();
    if (fl_param.partition == true && fl_param.mode != "centralized") {
        // 加载数据集
        dataset.load_from_file(model_param.path, fl_param);
        Partition partition;
        if (fl_param.partition_mode == "hybrid") {
            LOG(INFO) << "horizontal vertical dir";
            if (fl_param.mode == "horizontal")
                CHECK_EQ(fl_param.n_verti, 1);
            if (fl_param.mode == "vertical")
                CHECK_EQ(fl_param.n_hori, 1);
            partition.horizontal_vertical_dir_partition(dataset, n_parties, fl_param.alpha, feature_map, subsets,
                                                        fl_param.n_hori, fl_param.n_verti);
        } else if (fl_param.partition_mode == "vertical") {
            dataset.csr_to_csc();
            partition.homo_partition(dataset, n_parties, false, subsets, batch_idxs, fl_param.seed);
            for (int i = 0; i < n_parties; i++) {
                train_subsets[i] = subsets[i];
            }
        }else if (fl_param.partition_mode=="horizontal") {
            // 数据集横向分割
            dataset.csr_to_csc();
            fl_param.seed = 9;
            partition.homo_partition(dataset, n_parties, true, subsets, batch_idxs, fl_param.seed);
            for (int i = 0; i < n_parties; i++) {
                train_subsets[i] = subsets[i];
            }
        }
    }
    else if(fl_param.mode != "centralized"){
        for (int i = 0; i < n_parties; i ++) {
            subsets[i].load_from_file(model_param.paths[i], fl_param);
        }
        for (int i = 0; i < n_parties; i++) {
            train_subsets[i] = subsets[i];
        }
    }
    else{
        dataset.load_from_file(model_param.path, fl_param);
    }

    int n_test_dataset = model_param.test_paths.size();
    vector<DataSet> test_dataset(n_test_dataset);
    if (use_global_test_set) {
        if (model_param.reorder_label && fl_param.partition) {
            for (int i = 0; i < n_test_dataset; i++)
                test_dataset[i].label_map = dataset.label_map;
        }
        for (int i = 0; i < n_test_dataset; i++)
            test_dataset[i].load_from_file(model_param.test_paths[i], fl_param);
        if (model_param.reorder_label && fl_param.partition) {
            for (int i = 0; i < n_test_dataset; i++) {
                test_dataset[i].label = dataset.label;
                fl_param.gbdt_param.num_class = test_dataset[i].label.size();
            }
        }
    }

    fl_param.gbdt_param.objective = "binary:logistic";

    GBDTParam &param = fl_param.gbdt_param;
    //correct the number of classes
    if (param.objective.find("multi:") != std::string::npos || param.objective.find("binary:") != std::string::npos || param.metric == "error") {
        int num_class;
        if(fl_param.partition) {
            num_class = dataset.label.size();
            if ((param.num_class == 1) && (param.num_class != num_class)) {
                LOG(INFO) << "updating number of classes from " << param.num_class << " to " << num_class;
                param.num_class = num_class;
            }
        }
        if(param.num_class > 2)
            param.tree_per_round = param.num_class;
    }
    else if(param.objective.find("reg:") != std::string::npos){
        param.num_class = 1;
    }

    // 初始化参与方和服务器
    vector<Party> parties(n_parties);
    vector<int> n_instances_per_party(n_parties);
    Server server;
    if(fl_param.mode != "centralized") {
        LOG(INFO) << "initialize parties";
        for (int i = 0; i < n_parties; i++) {
            if(fl_param.mode == "vertical")
                parties[i].vertical_init(i, train_subsets[i], fl_param);
            else if(fl_param.mode == "horizontal" || fl_param.mode == "ensemble" || fl_param.mode == "solo")
                // 横向初始化参与方
                parties[i].init(i, train_subsets[i], fl_param);
            n_instances_per_party[i] = train_subsets[i].n_instances();
        }
        LOG(INFO) << "initialize server";
        if (fl_param.mode == "vertical") {
            if(fl_param.partition)
                server.vertical_init(fl_param, dataset.n_instances(), n_instances_per_party, dataset.y, dataset.label);
            else
                server.vertical_init(fl_param, train_subsets[0].n_instances(), n_instances_per_party, train_subsets[0].y, train_subsets[0].label);
        } else if (fl_param.mode == "horizontal" || fl_param.mode == "ensemble" || fl_param.mode == "solo") {
            server.horizontal_init(fl_param);
        }
    }

    // 开始训练
    LOG(INFO) << "start training";
    FLtrainer trainer;
    if (param.tree_method == "auto")
        param.tree_method = "hist";
    else if (param.tree_method != "hist"){
        LOG(INFO)<<"FedTree only supports histogram-based training yet";
        exit(1);
    }
    std::vector<float_type> scores;
    if(fl_param.mode == "hybrid"){
        LOG(INFO) << "start hybrid trainer";
        trainer.hybrid_fl_trainer(parties, server, fl_param);
        for(int i = 0; i < n_parties; i++){
            float_type score;
            if(use_global_test_set)
                score = parties[i].gbdt.predict_score(fl_param.gbdt_param, test_dataset[0]);
            //else
            //    score = parties[i].gbdt.predict_score(fl_param.gbdt_param, test_subsets[i]);
            scores.push_back(score);
        }
    }
    else if(fl_param.mode == "ensemble"){
        trainer.ensemble_trainer(parties, server, fl_param);
        float_type score;
        if(use_global_test_set) {
            score = server.global_trees.predict_score(fl_param.gbdt_param, test_dataset[0]);
            scores.push_back(score);
        }
    }
    else if(fl_param.mode == "solo"){
        trainer.solo_trainer(parties, fl_param);
        float_type score;
        for(int i = 0; i < n_parties; i++){
            if(use_global_test_set)
                score = parties[i].gbdt.predict_score(fl_param.gbdt_param, test_dataset[0]);
            //else
            //    score = parties[i].gbdt.predict_score(fl_param.gbdt_param, test_subsets[i]);
            scores.push_back(score);
        }
        float sum = std::accumulate(scores.begin(), scores.end(), 0.0);
        float sq_sum = std::inner_product(scores.begin(), scores.end(), scores.begin(), 0.0);
        float mean = sum / scores.size();
        float std = std::sqrt(sq_sum / scores.size() - mean * mean);
        LOG(INFO)<<"score mean (std):"<< mean << "(" << std << ")";
    }
    else if(fl_param.mode == "centralized"){
        GBDT gbdt;
        gbdt.train(fl_param.gbdt_param, dataset);
        float_type score;
        if(use_global_test_set) {
            score = gbdt.predict_score(fl_param.gbdt_param, test_dataset[0]);
            scores.push_back(score);
        }
    } else if (fl_param.mode == "vertical") {
        trainer.vertical_fl_trainer(parties, server, fl_param);
        float_type score;
        if(test_dataset.size() == 1) {
            score = parties[0].gbdt.predict_score_vertical(fl_param.gbdt_param, test_dataset[0], batch_idxs);
        }
        else {
            CHECK_EQ(fl_param.partition, 0);
            score = parties[0].gbdt.predict_score_vertical(fl_param.gbdt_param, test_dataset);
        }
        if(fl_param.joint_prediction)
            // score = server.predict_score_vertical(parties);
            score = 0.0;
        scores.push_back(score);
    }else if (fl_param.mode == "horizontal") {
        /* 横向训练
        */
        // 原始代码
        // LOG(INFO)<<"start horizontal training";
        // trainer.horizontal_fl_trainer(parties, server, fl_param);
        // LOG(INFO)<<"end horizontal training";
        // float_type score;
        // if(use_global_test_set)
        //     score = parties[0].gbdt.predict_score(fl_param.gbdt_param, test_dataset[0]);
        // //else
        // //    score = parties[0].gbdt.predict_score(fl_param.gbdt_param, test_subsets[0]);
        // scores.push_back(score);
        // LOG(INFO) << "score " << scores;

        /* Ext_SV
        */
        // std::vector<double> ext_sv(n_parties, 0);
        // std::vector<double> coef(n_parties, 0);
        // std::vector<int> set(n_parties);
        // std::iota(set.begin(), set.end(), 0);   // 生成 [0, 1, ..., n-1] 索引

        // // 计算每个子集大小的系数
        // for (int i = 0; i < n_parties; i++) {
        //     coef[i] = std::tgamma(i + 1) * std::tgamma(n_parties - i) / std::tgamma(n_parties + 1);
        // }

        // // 获取所有子集
        // std::vector<std::vector<int>> sets = power_set(set);

        // // 遍历所有子集并计算 Shapley 值
        // for (int idx = 0; idx < sets.size(); idx++) {
        //     LOG(INFO) << "set " << idx << " / " << sets.size();
        //     int s_size = sets[idx].size();
        //     std::vector<Party> selected_party(s_size);
        //     fl_param.n_parties = s_size;
        //     server.horizontal_init(fl_param);
        //     for (int k = 0; k < s_size; k++) {
        //         int _id = parties[sets[idx][k]].pid;
        //         DataSet _dataset = parties[sets[idx][k]].dataset;
        //         selected_party[k].init(_id, _dataset, fl_param);
        //     }
        //     double u = 0;
        //     trainer.horizontal_fl_trainer(selected_party, server, fl_param);
        //     u = selected_party[0].gbdt.predict_score(fl_param.gbdt_param, test_dataset[0]);

        //     // 对于子集中的每个玩家，累计贡献
        //     for (int i : sets[idx]) {
        //         ext_sv[i] += coef[s_size - 1] * u;
        //     }

        //     // 对于不在子集中的每个玩家，减去其贡献
        //     for (int i : set) {
        //         if (std::find(sets[idx].begin(), sets[idx].end(), i) == sets[idx].end()) {
        //             ext_sv[i] -= coef[s_size] * u;
        //         }
        //     }
        // }

        // double sum_sv = 0.0;
        // for (double& val : ext_sv) {
        //     sum_sv += val;
        // }
        // LOG(INFO) << "sum_sv = " << sum_sv; 

        /* MC_retrain
        */
        int local_m = 99;
        std::vector<double> sv(n_parties, 0.0);
        std::vector<int> idxs(n_parties);
        std::iota(idxs.begin(), idxs.end(), 0);
        std::mt19937 rng(9);
        std::vector<double> standalone_AUC(n_parties + 1, 0.0);
        double max_u = 0.0;

        auto t_start = timer.now();
        for (int i = 0; i < local_m; ++i) {  // 每个样本
            std::shuffle(idxs.begin(), idxs.end(), rng);
            LOG(INFO) << "\nsample " << i + 1 << "/" << local_m << " " << idxs;
            double old_u = 0;
            for(int j = 1; j <= n_parties; j++) {  // 样本增量计算
                std::vector<int> current_set(idxs.begin(), idxs.begin() + j);
                LOG(INFO) << "====== current set " << current_set << " ======";
                std::vector<Party> selected_party(current_set.size());
                fl_param.n_parties = current_set.size();
                server.horizontal_init(fl_param);
                for (int k = 0; k < current_set.size(); k++) {
                    int _id = parties[current_set[k]].pid;
                    DataSet _dataset = parties[current_set[k]].dataset;
                    selected_party[k].init(_id, _dataset, fl_param);
                }
                double temp_u = 0.0;
                // 如果有数据，取用旧数据
                if (j == 1 && standalone_AUC[current_set[0]] != 0.0){
                    temp_u = standalone_AUC[current_set[0]];
                }else if (j == n_parties && standalone_AUC[n_parties] != 0.0) {
                    temp_u = standalone_AUC[n_parties];
                }else {
                    // LOG(INFO) << "start horizontal training";
                    trainer.horizontal_fl_trainer(selected_party, server, fl_param);
                    // LOG(INFO) << "end horizontal training";
                    temp_u = selected_party[0].gbdt.predict_score(fl_param.gbdt_param, test_dataset[0]);
                }
                
                LOG(INFO) << "====== current_set AUC " << temp_u << " ======";
                // 记录数据
                if (j == 1 && standalone_AUC[current_set[0]] == 0.0) {
                    standalone_AUC[current_set[0]] = temp_u;
                }
                if (j == n_parties) {
                    if (i == 0) {
                        standalone_AUC[n_parties] = temp_u;
                        max_u = temp_u;
                    }else {
                        max_u = (temp_u + max_u) / 2;
                    }
                }
                // 截断
                // if (i > 0 && std::abs(max_u - temp_u) <= 0.001 && j != n_parties) {
                //     LOG(INFO) << "truncated!!!!!";
                //     break;
                // }
                double contribution = temp_u - old_u;
                sv[idxs[j - 1]] += contribution;
                old_u = temp_u;
            }
            // 记录小样本量的 Shapley 值
            if ((i+1) % 2 == 0) {
            // if ((i+1) % 2 == 0) {
                std::string outFileName_inner = "./exp_result/MC_retrain_error_a9a/MC_retrain_inner";
                outFileName_inner.append("_n_")
                                .append(std::to_string(n_parties))
                                .append("_m_")
                                .append(std::to_string(i + 1));
                std::ofstream outFile_inner(outFileName_inner);
                if (outFile_inner.is_open()) {
                    outFile_inner << "sv\t";
                    double sum_temp = 0;
                    for (const auto& val : sv) {
                        outFile_inner << val/(i+1) << ", ";
                        sum_temp += val/(i+1);
                    }
                    outFile_inner << "\n";
                    outFile_inner << "sum_sv\t" << sum_temp << "\n";
                    outFile_inner.close();
                } else {
                    LOG(ERROR) << "unable to write file";
                }
            }
        }
        double sum_sv = 0.0;
        for (double& val : sv) {
            val /= local_m;
            sum_sv += val;
        }
        LOG(INFO) << sv;
        LOG(INFO) << sum_sv;
        auto t_end = timer.now();
        std::chrono::duration<float> used_time = t_end - t_start;
        LOG(INFO) << "MC_SHAP using time:" << used_time.count() << " s";

        // 写文件
        std::string outFileName = "MC_retrain";
        outFileName.append("_n_")
                    .append(std::to_string(n_parties))
                    .append("_m_")
                    .append(std::to_string(local_m));
        std::ofstream outFile(outFileName);

        if (outFile.is_open()) {
            outFile << "sv\t";
            for (const auto& val : sv) {
                outFile << val << ", ";
            }
            outFile << "\n";
            outFile << "sum_sv\t" << sum_sv << "\n";
            outFile << "time\t" << used_time.count() << "s\n";

            outFile.close();
        } else {
            LOG(ERROR) << "unable to write file";
        }

        /* CC_retrain
        */
        // int local_m = 304;
        // std::vector<double> sv(n_parties, 0.0);
        // std::vector<int> idxs(n_parties);
        // std::iota(idxs.begin(), idxs.end(), 0);
        // std::mt19937 rng(9);
        // std::uniform_int_distribution<int> dist(1, n_parties);
        // // 初始化 utility 和 count 矩阵，大小为 (n+1)*n
        // std::vector<std::vector<double>> utility(fl_param.n_parties + 1, std::vector<double>(n_parties, 0));
        // std::vector<std::vector<int>> count(fl_param.n_parties + 1, std::vector<int>(n_parties, 0));

        // auto t_start = timer.now();
        // for (int i = 0; i < local_m; i++) {     // 每个样本
        //     std::shuffle(idxs.begin(), idxs.end(), rng);
        //     LOG(INFO) << "\nsample " << i + 1 << "/" << local_m << " " << idxs;
        //     int j = dist(rng);  // 分割点
        //     double u_1, u_2;
        //     LOG(INFO) << j;

        //     std::vector<int> set_1(idxs.begin(), idxs.begin() + j);
        //     std::vector<int> set_2(idxs.begin() + j, idxs.end());

        //     LOG(INFO) << "set_1" << set_1;
        //     LOG(INFO) << "set_2" << set_2;
        //     std::vector<Party> selected_party_1(set_1.size());
        //     std::vector<Party> selected_party_2(set_2.size());
            
        //     // selected_party_1训练
        //     fl_param.n_parties = set_1.size();
        //     server.horizontal_init(fl_param);
        //     for (int k = 0; k < set_1.size(); k++) {
        //         int _id = parties[set_1[k]].pid;
        //         DataSet _dataset = parties[set_1[k]].dataset;
        //         selected_party_1[k].init(_id, _dataset, fl_param);
        //     }
        //     trainer.horizontal_fl_trainer(selected_party_1, server, fl_param);
        //     u_1 = selected_party_1[0].gbdt.predict_score(fl_param.gbdt_param, test_dataset[0]);

        //     // selected_party_2训练
        //     if (j == n_parties) {
        //         u_2 = 0;
        //     } else {
        //         fl_param.n_parties = set_2.size();
        //     server.horizontal_init(fl_param);
        //     for (int k = 0; k < set_2.size(); k++) {
        //         int _id = parties[set_2[k]].pid;
        //         DataSet _dataset = parties[set_2[k]].dataset;
        //         selected_party_2[k].init(_id, _dataset, fl_param);
        //     }
        //     trainer.horizontal_fl_trainer(selected_party_2, server, fl_param);
        //     u_2 = selected_party_2[0].gbdt.predict_score(fl_param.gbdt_param, test_dataset[0]);
        //     }
            
        //     LOG(INFO) << "u_1 " << u_1;
        //     LOG(INFO) << "u_2 " << u_2;

        //     // 更新效用和计数
        //     std::vector<int> temp(n_parties, 0);
        //     for (int k = 0; k < j; k++) {
        //         temp[idxs[k]] = 1;
        //     }
        //     for (int k = 0; k < n_parties; k++) {
        //         utility[j][k] += temp[k] * (u_1 - u_2);
        //         count[j][k] += temp[k];
        //     }
        //     std::fill(temp.begin(), temp.end(), 0);
        //     for (int k = j; k < n_parties; k++) {
        //         temp[idxs[k]] = 1;
        //     }
        //     for (int k = 0; k < n_parties; k++) {
        //         utility[n_parties - j][k] += temp[k] * (u_2 - u_1);
        //         count[n_parties - j][k] += temp[k];
        //     }

        //     // 记录小样本 Shapley 值
        //     // if ((i + 1) % 100 == 0) {
        //     if ((i + 1) % 2 == 0) {
        //         std::vector<double> sv_inner(n_parties, 0);
        //         double sum_sv_inner = 0;
        //         for (int k = 1; k <= n_parties; k++) {
        //             for (int p = 0; p < n_parties; p++) {
        //                 if (count[k][p] == 0) {
        //                     sv_inner[p] += 0;
        //                 } else {
        //                     sv_inner[p] += (utility[k][p] / count[k][p]);
        //                 }
        //             }
        //         }
        //         for (double & val : sv_inner) {
        //             val /= n_parties;
        //             sum_sv_inner += val;
        //         }
        //         std::string outFileName_inner = "./exp_result/CC_retrain_time_a9a/CC_retrain_inner";
        //         outFileName_inner.append("_n_")
        //                         .append(std::to_string(n_parties))
        //                         .append("_m_")
        //                         .append(std::to_string(i + 1));
        //         std::ofstream outFile_inner(outFileName_inner);
        //         if (outFile_inner.is_open()) {
        //             outFile_inner << "sv\t";
        //             for (const auto& val : sv_inner) {
        //                 outFile_inner << val << ", ";
        //             }
        //             outFile_inner << "\n";
        //             outFile_inner << "sum_sv\t" << sum_sv_inner << "\n";
        //             outFile_inner.close();
        //         } else {
        //             LOG(ERROR) << "unable to write file";
        //         }
        //     }
        // }   // end 样本

        // LOG(INFO) << "utility " << utility;
        // LOG(INFO) << "count " << count;

        // for (int i = 1; i <= n_parties; i++) {
        //     for (int j = 0; j < n_parties; j++) {
        //         if (count[i][j] == 0) {
        //             sv[j] += 0;
        //         }else {
        //             sv[j] += (utility[i][j] / count[i][j]);
        //         }
        //     }
        // }
        // double sum_sv = 0.0;
        // for (double & val : sv) {
        //     val /= n_parties;
        //     sum_sv += val;
        // }

        // LOG(INFO) << sv;
        // LOG(INFO) << sum_sv;
        // auto t_end = timer.now();
        // std::chrono::duration<float> used_time = t_end - t_start;
        // LOG(INFO) << "CC_SHAP using time: " << used_time.count() << " s";

        // std::string outFileName = "./exp_result/CC_retrain_time_a9a/CC_retrain";
        // outFileName.append("_n_")
        //             .append(std::to_string(n_parties))
        //             .append("_m_")
        //             .append(std::to_string(local_m));
        // std::ofstream outFile(outFileName);

        // if (outFile.is_open()) {
        //     outFile << "sv\t";
        //     for (const auto& val : sv) {
        //         outFile << val << ", ";
        //     }
        //     outFile << "\n";
        //     outFile << "sum_sv\t" << sum_sv << "\n";
        //     outFile << "time\t" << used_time.count() << "s\n";

        //     outFile.close();
        // } else {
        //     LOG(ERROR) << "unable to write file";
        // }

        
        /* CCN
        */
        // int n_samples = 20;
        // std::vector<double> sv(n_parties, 0);
        // std::vector<std::vector<double>> var(n_parties, std::vector<double>(n_parties));
        // std::vector<int> idxs(n_parties);
        // std::iota(idxs.begin(), idxs.end(), 0);
        // std::mt19937 rng(9);
        // std::uniform_int_distribution<int> dist(1, n_parties);


        /* MC_split
        */
        // LOG(INFO)<<"start horizontal training";
        // trainer.horizontal_fl_trainer_mc(parties, server, fl_param);
        // LOG(INFO)<<"end horizontal training";
        // float_type score;
        // score = parties[0].gbdt.predict_score(fl_param.gbdt_param, test_dataset[0]);
        // scores.push_back(score);
        // LOG(INFO) << "score " << scores;

        /* CC_split
        */
        // LOG(INFO)<<"start horizontal training";
        // trainer.horizontal_fl_trainer_cc(parties, server, fl_param);
        // LOG(INFO)<<"end horizontal training";
        // float_type score;
        // score = parties[0].gbdt.predict_score(fl_param.gbdt_param, test_dataset[0]);
        // scores.push_back(score);
        // LOG(INFO) << "score " << scores;

        // LOG(INFO)<<"START HORIZONTAL TRAINING";
        // trainer.horizontal_fl_trainer_streamline(parties, server, fl_param);
        // LOG(INFO)<<"END HORIZONTAL TRAINING";
        // float_type score;
        // score = parties[0].gbdt.predict_score(fl_param.gbdt_param, test_dataset[0]);
        // scores.push_back(score);
        // LOG(INFO) << "score " << scores;
    }
    parser.save_model(fl_param.gbdt_param.model_path, fl_param.gbdt_param, server.global_trees.trees);
    return 0;
}
