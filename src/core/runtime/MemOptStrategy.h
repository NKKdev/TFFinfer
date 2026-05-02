//
// Created by nkk on 2/10/26.
//

#ifndef TFFINFER_MEMOPTSTRATEGY_H
#define TFFINFER_MEMOPTSTRATEGY_H
#include "ModuleFactory.h"
#include "ModuleObject.h"
#include "graph/GraphNode.h"
namespace tff::core::runtime {
#define FOR_EACH_STRATEGY(X) \
X(TFF_MEM_OPT_STRATEGY_DEFAULT, "default")\
X(TFF_MEM_OPT_STRATEGY_UNKNOWN, "unknown")

    /**
     *
     * @brief 优化策略类型
     */
    enum MemOptStrategyType {
#define DEFINE_ENUM(name, type_str) name,
        FOR_EACH_STRATEGY(DEFINE_ENUM)
#undef DEFINE_ENUM
        TFF_MODEL_ARCH_COUNT
    };

    constexpr const char *to_string(const MemOptStrategyType arch_type) {
#define CASE_STR(name, str) case MemOptStrategyType::name: return str;
        switch (arch_type) {
            FOR_EACH_STRATEGY(CASE_STR)
            default: return "invalid";
        }
#undef CASE_STR
    }

    constexpr MemOptStrategyType from_string(std::string_view enum_str) {
#define CASE_STR(name, str) if (enum_str == str) return MemOptStrategyType::name;
        FOR_EACH_STRATEGY(CASE_STR)
#undef CASE_STR
        return MemOptStrategyType::TFF_MEM_OPT_STRATEGY_UNKNOWN;
    }
    /**
     *
     * @brief 内存优化策略
     */
    class MemOptStrategyBase: public module::ModuleObject {
    public:
        MemOptStrategyBase()= default;
        ~MemOptStrategyBase() override = default;
    public:
        /**
         *
         * @brief 获取内存优化策略对象
         * @param type
         * @return
         */
        inline static std::shared_ptr<MemOptStrategyBase> get_mem_opt_strategy(MemOptStrategyType type = TFF_MEM_OPT_STRATEGY_DEFAULT) {
            return std::dynamic_pointer_cast<MemOptStrategyBase>(factory::ModuleFactory::instance()
                ->create_shared<tff::module::ModuleObject>(MEM_OPT_STRATEGY_FLAG,
                tff::factory::ModuleKeyType(to_string(TFF_MEM_OPT_STRATEGY_DEFAULT))));
        }
    public:
        /**
         *
         * @brief 内存优化
         * @param node
         * @return
         */
        virtual std::shared_ptr<core::graph::GraphNode> mem_optimize(const std::shared_ptr<graph::GraphNode> &node);
        /**
         *
         * @brief 是否需要优化
         * @param node
         * @return
         */
        virtual bool should_optimize(const std::shared_ptr<graph::GraphNode> &node);

    };
}
#endif //TFFINFER_MEMOPTSTRATEGY_H