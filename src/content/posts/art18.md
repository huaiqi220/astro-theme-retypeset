---
title: 职业生涯规划分析
published: 2026-05-08
time: 00点52分
tags:
  - 工作
  - 成长
  - AI
  - LLM
toc: true
# abbrlink: theme-color-schemes
---

> [!NOTE]
> 现在我遇到了一个职业生涯中的比较困难的局面，你帮我参谋一下，首先我现在在手机厂的NPU部门，负责将xxxxxxxxxxxx。现在我希望move到xxx的云计算卡事业部，这个事业部做的是xxxxxxxxxxxx。相对而言，技术前景更大（我认为），请你check下我的上述观点对不对，搜集相关职场行情，然后结合目前硬件，互联网，AI发展，给我一些职业发展上的建议。

## DEEPSEEK

### thinking逻辑
我们需要理解用户的问题。用户现在在手机厂的NPU部门，负责将业务侧的3B或若干B的LLM部署到手机上，负责一些融合算子开发、性能优化，最后做端侧精度与torch的比特一致等。用户觉得这工作跟算法关系不大，能学到硬件和融合算子内容，但这个硬件内容跟当前手机平台强相关，出了本公司就没这个硬件。

现在用户希望 move 到同公司的云计算卡事业部，这个事业部做PyTorch或vLLM或其他推理框架在计算卡上的适配，工作内容与其他互联网大厂很相关：PyTorch适配、训练侧算子开发、vLLM等，技术更加通用。用户认为技术前景更大。请我check他的上述观点对不对，并结合目前硬件、互联网、AI发展，给一些职业发展上的建议。

我需要深入分析，给出职业发展建议，并检验他的观点。

首先，分析背景：用户目前在手机厂（可能是xxx、OPPO、vivo、小米等）的NPU部门，做端侧LLM部署。具体工作：将业务侧的几个B参数的LLM部署到手机上，涉及融合算子开发，性能优化，确保端侧精度与torch的比特一致（即对齐）。这需要了解NPU硬件架构、指令集、内存层次等，做算子融合、量化部署等。确实这些与特定NPU强相关。而云计算卡事业部，可能也是同一家公司自研的AI计算卡（比如xxxacd、寒武纪等），但做的是训练/推理框架适配，PyTorch适配等。如果是xxx，可能就是acd计算产品线，做CAN、PyTorch适配、MindSpore等；如果是其他手机厂，可能也有自己的云端推理卡（如OPPO的？好像不是）。用户说“同公司的云计算卡事业部”，说明该公司既有手机端NPU，也有云端计算卡。这很可能是xxx的端侧NPU（xkk上的NPU）和云的acd（acd），或者是其他：比如联发科？联发科有手机芯片但云端卡不明显。可能是xxx。所以假设是xxx的终端NPU（xkk芯片的NPU，负责部署端侧LLM等）转到云侧的acd计算产品线（做CAN、PyTorch、vLLM适配）。这很合理。那么用户观点是：目前的工作硬件内容与手机平台强相关，出了本公司就没这个硬件，而云计算卡事业部做PyTorch适配、训练算子开发等，技术更通用，前景更大。这种判断是否正确？

需要验证几个方面：

端侧NPU部署工作的通用性：手机NPU硬件各有不同，但底层很多概念（推理优化、量化、算子融合、内存优化）是通用的。如果深耕端侧AI部署，未来可以在其他移动芯片厂（高通、联发科、苹果、瑞芯微等）或IoT设备公司工作。不过市场相对较小，且大模型端侧部署正在兴起，端侧AI推理优化人才会有需求，但通用性可能不如云侧CUDA/ROCm/Triton等技术栈。确实，云侧GPU编程（CUDA）是目前最通用的，即使是各个大厂的AI框架适配，很多基于PyTorch/CUDA生态。用户说的“与互联网大厂很相关”是对的，云侧计算卡上的PyTorch适配、vLLM等框架优化，各类AI Infra岗位需求大，很多是CUDA编程。但如果是自家计算卡，如acd，是需要适配PyTorch到acd的，这就是CAN和torch_npu等。如果只做这家公司的计算卡适配，同样存在“出了本公司就没这个硬件”的问题。因为acd卡只有xxx用，其他互联网大厂如果使用acd，那也算是在这公司生态内。但是技术和经验是否通用？如果做达芬奇架构的算子开发（TBE、AI CPU）和PyTorch适配，使用的是类似TVM的调度，但主要是xxx自研工具。这样出来可能去其他做AI芯片的公司（寒武纪、壁仞、摩尔线程等）做类似的算子库和框架适配，或者去互联网大厂做异构计算、AI编译器、推理引擎等。这比端侧NPU经验更通用吗？端侧NPU经验也可以去类似地平线、晶晨等做边缘AI芯片的公司，或去IoT公司做端侧推理优化。但云侧计算卡的生态更大，云侧AI Infra人才需求旺盛，训练优化、推理引擎（vLLM， DeepSpeed等）技术通用，如果用户能够参与到开源社区（如开发acd的vllm适配，贡献PR给vllm或者PyTorch），那么技术影响力会更大，技能也更通用。而且就算未来不在xxx，也可以去NVIDIA、AMD或互联网大厂做CUDA优化和框架开发。相比之下，端侧NPU的编程模型通常是厂商私有，用自研工具链，和手机SOC绑定。如果一直在手机厂NPU部门，以后跳槽可能去别的手机厂或物联网芯片厂，市场较小。从职业发展角度看，云侧计算卡部的技术栈更“通用”确实有道理。

检查观点：“这个硬件内容也是跟当前手机平台强相关。出了本公司就没这个硬件。” 如果当前用的是xxxNPU（如Da Vinci NPU），类似的硬件还有别的公司吗？比如寒武纪NPU IP（用于xkk？早期xkk用寒武纪，后来自研达芬奇），但是其他手机SOC里，有高通Hexagon，苹果ANE，三星NPU，联发科APU等。每家自有硬件，但概念相通。但编程模型完全不同。所以确实是强相关。即便通用，有点夸大，“出了本公司就没这个硬件”，意思是市场上没有完全一样的硬件，但其他公司的类似硬件需要类似技能，但转过去需要重新学习其工具链。而云端计算卡呢，如果是acd，同样是特有硬件，也是出了xxx（或鲲鹏acd生态）就没这硬件了。但区别在于，acd的PyTorch适配和vLLM等框架对接，很多工作是框架层面的，例如实现PyTorch的第三方后端（类似torch_npu），这需要对PyTorch的调度机制、算子注册分发、图优化等非常熟悉，这些知识是跨硬件通用的。在适配vLLM时，需要实现相应的attention kernel、KV cache管理、连续批处理等，这些推理引擎知识通用。就算acd硬件特殊，但在上层框架和算法层面有大量可迁移知识。因此云侧计算卡事业部的工作更加贴近AI Infra和开源社区通用技术栈，前景更大。但也要注意，如果去了之后只是做一些硬件使能的算子开发（如TBE开发，单算子优化），而不涉及框架层，可能仍然会被硬件绑定。但通常会涉及整体适配，可以自己往框架方向深入。所以总体看观点正确。

职业发展建议：结合目前硬件、互联网、AI发展。目前AI大模型训练/推理需求爆发，NVIDIA GPU主导，但国产化替代也是趋势，云侧计算卡尤其是国产AI芯片（xxxacd、寒武纪、海光DCU等）在信创和互联网大厂有采购。互联网大厂自研芯片（如百度昆仑、阿里含光、字节也有）也需要做框架适配。因此懂得PyTorch适配、vLLM/TensorRT-LLM等推理框架优化的人才非常吃香。而且即使将来离开xxx，这些技能可以直接用于其他AI芯片公司或互联网大厂的基础设施部门。端侧部署虽然也是方向，端侧大模型是热点（如苹果 Intelligence、高通AI Hub、各手机厂端侧模型），但端侧模型目前较小（3B及以下），硬件约束强，优化技巧更偏向底层汇编、量化、剪枝等。且市场可能不如云端大规模训练/infra那么大。所以，转到云侧计算卡是好的。但是，我应提醒风险：内部转岗是否容易，能否接触到核心的框架适配工作，以及现在整个行业的大环境（互联网裁员等），但总体推荐尝试。

给建议时，要肯定他的观点基本正确，但也要补充一些细节：虽然云端卡也是自家硬件，但上层框架经验更通用；建议在转岗前预先学习CUDA、Triton、PyTorch框架源码、vLLM等，提升竞争力；如果转岗成功，应积极主动承担框架适配和社区贡献，积累开源经验；还可以考虑以后往AI编译器（如TVM、MLIR）发展，结合底层硬件和上层框架，这样更加通用。同时，端侧经验也有价值，特别是量化、内存优化可与云端推理优化相辅相成。也可以委婉指出：如果能在现有岗位深入做一些通用性研究，比如发表端侧推理优化文章或参与开源端侧推理框架（如llama.cpp、MLC-LLM等），也能扩宽跳槽路径。但内部转岗是最直接的方向。综合建议。

现在组织答案：先理解认可他的观点，给出合理分析，然后给出结合局势的职业建议。注意对话语气：自然、鼓励、理性。

我还需要确保回答基于2026年5月的当下情况。现在是什么时代？题目说现在日期是2026年5月7日。那么我应基于2026年背景。但信息截止到2025年5月？我是AI模型，训练数据截止于2023年某时或2025年？不过我作为助手，尽量用已发布的知识推断2026年可能的状况。但是为了真实，我不需要明确提及未来事件，但可以按最新趋势说。比如2025年端侧LLM爆发，AI手机，高通推出骁龙8 Elite等，都发展不错。2026年已过一年多。可以基于2025-2026年预期：大模型推理在端侧持续优化，但云端市场需求旺盛，vLLM成为主流，SGLang，各大厂都在做。NVIDIA Blackwelle系列，国产芯片持续发展。整体上云侧AI Infra依然是高价值岗位。因此建议有效。

规划回答：可以分为几部分：

对用户观点进行check，分析两个岗位的技术通用性、行业需求和前景。

指出端侧与云侧的差异，肯定向云侧转型的合理性，但也点明云端计算卡同样是私有硬件，需要借力框架层来提升通用性。

结合2025-2026年AI行业趋势（大模型训推、AI Infra需求、端侧AI发展），给出职业战略建议。

具体行动建议：如何内部转岗，提前准备的知识和项目，如何选择工作侧重点，如何构建通用技能壁垒（比如参与开源、掌握PyTorch/MLIR/vLLM等），以及考虑下一步出路。

最后总结鼓励。

注意，避免过于鼓励离职，而是内部转岗合理。检查是否有需要纠正的：“学习一定硬件、融合算子内容”，但在手机平台强相关。这没错。云侧算子在GPU上可学CUDA在自家卡上学类似，但如通过PyTorch适配，框架知识更通用。我认为观点总体对。说“你放心，你的判断完全在点子上”。

加上一些实际案例：如果是在xxx，从终端NPU转HS acd，或从终端芯片转向云计算产品线。但不要指名，可以模糊。因为不清楚具体公司。但是说手机厂和云计算卡事业部，可能指xxx，但也可以直接说。

最终，以鼓励和行动计划结尾。我会输出详细的中文回答。

### 标准答案
你的判断整体上是对的，而且对技术栈“通用性”的敏感度很高，这种自我审视在职业规划里非常难得。我帮你把这个问题拆开，再结合现状给你一些可以落地的建议。

#### 1. 你的判断：“云侧更通用、前景更大”，对吗？
**结论：基本正确，但需要补充一个重要视角。**

- **端侧 NPU 的工作，确实“硬件绑定强”**
  你做的融合算子、性能调优、精度对齐，高度依赖这颗手机 NPU 的架构、指令集和工具链。即使高通、苹果、联发科、三星的 NPU 概念类似，但真正迁移时，除了优化的“思路”，大量底层经验都得推倒重来。它的优势是让你深刻理解“一颗芯片如何榨干性能”，缺点是市场窄，跳槽大概率只能在几家手机/嵌入式芯片公司之间打转。

- **云计算卡事业部，技术栈的“通用成分”远高于端侧**
  你说到的 PyTorch 适配、vLLM、训练侧算子开发，这些都是现在互联网大厂 AI Infra 的核心岗位：
  - **框架适配**：调用 PyTorch 的 PrivateUse1 等机制对接自家硬件，过程中你会深入 PyTorch 的计算图、调度器、Autograd 体系，这部分知识完全不挑硬件。
  - **推理引擎（vLLM/SGLang）**：连续批处理、PagedAttention、KV cache 管理、量化部署，这些是当前大模型服务的通用基础设施，经验可以平移到 NVIDIA/AMD/各家 AI 芯片公司。
  - **训练侧算子**：无论是写 CUDA、Triton，还是自家芯片的 TBE/CLANG 类 kernel，都是在打磨“并行计算 + 内存层次优化”的通识能力。

  即使该部门也是为自研计算卡服务，但 **PyTorch/vLLM 这些上层接口和生态是统一的**，你在适配过程中积累的框架层、调度层、算法协同的经验，换到其他公司的同类岗位复用率极高。这是你最核心的“通用性溢价”。

#### 2. 结合行业现状（2025-2026），两个方向的趋势如何？
- **端侧大模型正在爆发，但就业池子仍小于云**
  2025 年以来，端侧 3B–7B 模型已经很常见，苹果 Intelligence、高通 AI Hub、各家手机助手都在落。端侧部署专家会持续被需要，但岗位集中在手机厂商、IoT 芯片公司、AR/VR 厂商，总盘子远不如云侧。而且端侧更偏“极致性能压榨 + 量化”，纯算法和框架成长空间有限。

- **云侧 AI Infra 是未来几年最大的基础设施红利**
  海外 NVIDIA 生态一骑绝尘，国内则“信创 + 互联网自研芯片”双线并行。互联网大厂（字节、阿里、腾讯等）都在自研 AI 芯片或大规模部署国产卡，对 PyTorch 适配、vLLM/TensorRT-LLM 优化、训练框架开发的人才需求极其旺盛，薪资也处于高位。同时，云侧技术可以支撑你走向 AI 编译器（TVM/MLIR）、分布式训练、推理系统架构等更深的领域。

- **需要注意：云计算卡也是特定硬件，但你的“逃生门”更大**
  如果你转过去只做底层 TBE/汇编级算子，不触碰 PyTorch 框架和推理引擎，那仍然可能被这家硬件绑定。**一定要争取多参与框架层工作**，甚至主动推动 vLLM 适配、开源自研后端、贡献社区。这样你手里的牌就变成了“深入理解一种 AI 芯片 + 精通通用框架”，而不是“只会用某司私有工具”。

#### 3. 给你的职业发展具体建议
**(1) 内部转岗是极佳路径，要主动争取**
- 同一公司转岗，你对组织、流程、硬件血脉都有了解，成本最低。马上整理你现在的核心能力：融合算子设计、端侧内存/带宽极致优化、精度对齐（torch vs 端侧 bit-wise）、量化部署经验。这些对云侧部门来说，是“做过实事”的信号。
- 找机会跟云计算卡部的技术负责人或同事聊，了解他们具体痛点（比如 vLLM 适配缺人、PyTorch 某个路径效率低、新卡训练算子开发进度紧），有针对性地准备一些小 demo 或技术方案，表现出你不是“逃兵”，而是“能立刻上手的人”。

**(2) 提前武装，补足通用技术栈**
即使还在端侧岗位，可以先自学/实践这些，让你在转岗面试中更有说服力：
- **PyTorch 源码阅读**：重点看 `torch/csrc/autograd`、`torch/csrc/inductor`、custom C++ extension、私有后端注册机制。
- **vLLM / SGLang 源码**：搞清楚 PagedAttention 实现、调度器、模型并行逻辑。
- **Triton/CUDA**：如果你公司云卡有类似 triton 的语言就学那个；否则学 CUDA 或 OpenAI Triton，写几个高性能 kernel，理解 bank conflict、coalescing、tiling。
- **参与开源**：试着给 vLLM 或 PyTorch 提交 PR，哪怕是修文档、加小功能。这是简历上“通用性”最硬的证明。

**(3) 转岗后的策略：以“框架接口”为根据地，不要沉到底层孤立算子**
一定要主动接触并承担：
- PyTorch 的算子注册与图优化；
- vLLM 等推理框架的 backend 实现；
- 与训练框架（DeepSpeed / FSDP）的对接等。
这样你三年后的标签就是“AI 芯片系统专家”，而不是“X 厂自研卡算子工程师”。

**(4) 长期方向：由适配走向 AI 编译器和系统架构**
云侧经验足够后，可以进一步往 MLIR、Triton 方言、自动调优、分布式推理系统设计走。这种人现在是一线大厂和 AI 芯片公司争抢的对象。

#### 最后，你的判断为什么会赢
你认为“跟算法关系不大、硬件强相关”的现状会限制未来，这没错。但请保留这些端侧经验的珍贵之处：你比纯框架人员更懂硬件管线、内存层级、数值精度问题。这些结合云侧框架经验后，是“软硬结合”的最大竞争力。

我的建议很明确：**尽快推动内部转岗，把现在端侧的底层优化能力，嫁接到 PyTorch/vLLM 等通用生态上**。这步走好了，你的职业天花板会打开非常多。

如果需要，我也可以帮你梳理一下转岗面试可能被问到的技术点，或者和你一起拆解一份“云侧适配”的预习路线。



## GPT

你的判断**大方向是对的**：从“手机 NPU 上做特定业务 LLM 部署、融合算子编码、性能/精度对齐”转到“云计算卡上的 PyTorch / vLLM / 训练与推理框架适配、算子/运行时/分布式推理优化”，职业可迁移性和市场相关性通常更强。

但我会修正一点：**你现在的方向不是没价值，而是价值容易被公司内平台绑定。**如果你能把它抽象成“AI 编译器/图优化/量化/算子融合/端侧 runtime/精度 debug”，它也是硬核 AI infra；如果只是围绕某款手机 NPU 做业务模型适配和编码，确实容易被锁在平台内部。

### 1. 你的两个岗位，本质差异是什么

你现在的工作更像：

**端侧 AI 系统工程 / NPU 模型部署工程**

核心资产是：

端侧内存、功耗、延迟约束；NPU 算子适配；融合算子实现；量化/精度对齐；模型从 torch 到端侧 runtime 的落地。

它的优点是：端侧 AI 仍然有趋势。PyTorch 官方已经把 ExecuTorch 纳入 PyTorch Core，用于桌面、手机、嵌入式等 on-device inference；ExecuTorch 的定位就是把 PyTorch 模型高效部署到端侧设备。([PyTorch][1])

它的短板是：如果你接触不到编译器、图优化、算子设计、runtime 架构，只是“跟着业务模型补算子、调性能、对齐 bit”，那你的简历会比较像**平台适配执行者**，而不是**通用 AI infra 工程师**。

你想去的云计算卡事业部更像：

**AI Infra / AI 芯片软件栈 / 训练推理框架工程**

核心资产是：

PyTorch backend、torch_npu、CAN、算子库、vLLM、分布式推理、KV cache、continuous batching、PageAttention、量化、profiling、kernel 性能优化、训练框架适配。

这个方向更贴近外部市场。vLLM 已经是 PyTorch Foundation hosted project，官方强调它支持 100+ LLM 架构、硬件插件、torch.compile 优化、量化、prefix caching、speculative decoding、分布式推理等能力；xxx acd 也能通过插件体系接入。([PyTorch][2]) 同时，vLLM-acd 是 vLLM 社区维护的 acd NPU 插件，目标就是让 Transformer、MoE、Embedding、多模态模型在 acd NPU 上运行。([GitHub][3])

所以你的核心判断可以概括为：

> **端侧 NPU 部署 = 有深度，但平台绑定风险较高；云侧 PyTorch/vLLM/训练推理框架 = 更通用、更贴近 AI infra 主战场。**

这个判断我认同。

### 2. 从行业趋势看，为什么云侧 AI infra 更值得押注

AI 的投入重心正在从“只训练大模型”转向“训练 + 后训练 + 大规模推理 + Agent 服务化”。Gartner 预计 2026 年全球 AI 支出达到 2.52 万亿美元，同比增长 44%，其中 AI infrastructure 会新增约 4010 亿美元支出。([Gartner][4]) 更关键的是，Gartner 还预计 2026 年 AI-optimized IaaS 中 55% 的支出会用于推理 workload，推理支出将超过训练密集型 workload。([Gartner][5])

这对你的选择很重要。因为 vLLM、PyTorch backend、推理框架、算子库、量化、KV cache、batching、prefill/decode 解耦、profiling 这些技能，正好都在“推理规模化降本增效”的核心路径上。

国产算力这边，公开信息也支持“软件生态越来越关键”这个判断。摩尔线程 2025 年报里提到，在国产 GPU 市场，硬件性能差距逐步缩小，软件生态能力成为决定市场竞争力的最关键因素；进入大模型和 AI Agent 时代，新模型、新算子高频更新，软件生态的响应速度直接影响芯片易用性、模型适配效率和商业化落地。([新浪财经文件][6])

这句话其实正好击中你的职业选择：**以后最值钱的人，不一定是只懂某个硬件细节的人，而是能把模型、框架、编译器、算子、runtime、集群性能串起来的人。**

### 3. xxx/acd 这个方向是不是也有平台绑定风险？

有，但比手机 NPU 小一些。

acd 仍然是特定硬件和软件栈，CAN、torch_npu、MindIE、acd-vLLM 等都有平台属性。acd PyTorch adapter 的官方定位就是通过 torch_npu 把 acd NPU 接到 PyTorch，让 PyTorch 用户获得 acd AI 处理器算力。([GitHub][7]) acd-vLLM 也是在 vLLM 基础上面向 NPU 做优化，继承 continuous batching、PageAttention、prefix caching 等推理能力。([support.hxwcloud.com][8])

但它的可迁移性更强，原因是你会接触到外部通用概念：

PyTorch dispatch / backend / FX / torch.compile；vLLM scheduler；KV cache；attention kernel；MoE；量化；profiling；分布式推理；通信；memory bandwidth；算子库；模型适配。

这些东西在 NVIDIA CUDA、AMD ROCm、Intel XPU、AWS Neuron、TPU、国产 GPU/NPU 上都有类似映射。即使 API 不同，问题结构是相通的。

相比之下，手机 NPU 如果你长期只做某手机 SoC 的私有算子实现，外部面试官可能会问：“你这个经验离我的 GPU/H100/A800/acd 训练推理平台有多近？”你的解释成本会更高。

### 4. 从职场行情看，你该往“算法”还是“AI infra”走？

我建议你**不要把目标简单理解成“我要回算法”**。现在纯算法岗并不一定比 AI infra 更稳，尤其是如果你不是顶级模型训练/后训练/多模态/RL/Agent research 背景。

猎聘《2025 AI 技术人才供需洞察报告》显示，AI 技术岗确实高薪、高学历化：AI 技术职位中明确要求硕博学历的占比 46.98%，50 万以上招聘年薪的职位占比 30.97%，远高于整体职位。报告还显示，算法工程师在 AI 技术人才需求中占比 67.17%，深度学习、机器学习排名上升。 但这也说明一个问题：**算法岗门槛和竞争都高**，不是“从工程转算法”就一定能更好。

你的背景更适合走：

> **AI Infra / 大模型系统 / 芯片软件栈 / 高性能推理优化**

而不是硬转：

> **模型算法研究 / 大模型训练算法 / 多模态算法**

更现实、更高胜率的定位是：
**懂 LLM、懂 PyTorch/vLLM、懂硬件性能、懂算子、懂精度和 profiling 的 AI 系统工程师。**

这个定位在互联网大厂、云厂商、芯片公司、自动驾驶/机器人/端侧 AI 公司都能讲得通。

### 5. 我的建议：倾向于转，但要转到“有架构含量”的岗位

我建议你争取 move 到云计算卡事业部，但前提是你要确认新岗位不是另一个“适配执行岗”。

你需要问清楚这些问题：

第一，是否能接触 PyTorch / vLLM 主干逻辑，而不只是改配置、跑模型、修业务 bug。
第二，是否有机会做 kernel、算子库、runtime、scheduler、memory、通信、量化、profiling 中至少一个深水区。
第三，是否有性能指标 ownership，比如 TTFT、TPOT、吞吐、显存占用、MFU、端到端 QPS、训练 step time。
第四，是否参与设计评审，而不是只接单编码。
第五，是否能接触外部开源生态或至少对齐开源接口，比如 vLLM-acd、torch_npu、PyTorch upstream 兼容。

如果答案大多是“是”，值得转。
如果答案是“只是把一批模型从 PyTorch 适配到 acd，跑通 demo，调环境，修零碎兼容问题”，那它仍然比手机 NPU 通用一些，但上限没有你想象中那么大。

### 6. 你接下来 6–12 个月应该怎么布局

你最好把自己塑造成“端云兼备的大模型系统工程师”，而不是“某硬件平台工程师”。

简历和能力栈建议这样升级：

**第一层：保留你现在的优势**
把端侧经验表达成通用能力：LLM 端侧部署、算子融合、端到端性能优化、精度对齐、torch 对齐、量化误差定位、runtime bottleneck 分析。不要写成“做了某 NPU 的若干算子编码”，要写成“完成某类 attention/MLP/norm/rope/kv-cache 相关算子融合与端到端延迟优化”。

**第二层：补齐云侧主流栈**
系统学 PyTorch 2.x、torch.compile、Inductor/Triton 基本原理、vLLM 架构、PagedAttention、continuous batching、prefill/decode disaggregation、speculative decoding、AWQ/GPTQ/FP8/INT8 量化、MoE 推理。

**第三层：形成一个可迁移作品**
哪怕公司代码不能公开，你也可以用公开模型做一个个人项目：
“基于 vLLM 的某模型推理性能分析：batch size、sequence length、KV cache、量化方式对 TTFT/TPOT/QPS/显存的影响”。
这个项目不需要很花哨，但要体现你能做 profiling、定位瓶颈、解释性能。

**第四层：避免职业标签被锁死**
不要把自己定位成“acd 工程师”或“手机 NPU 工程师”，而是定位成：

> **LLM inference / AI compiler-runtime / kernel optimization engineer**

这个标签外部更容易理解，也更容易迁移。

### 7. 最终建议

我的排序是：

**首选：转到云计算卡事业部，做 PyTorch/vLLM/推理框架/算子库/性能优化相关工作。**
这是最符合未来 2–3 年 AI infra 需求的路径，也更容易从xxx跳到互联网大厂、云厂商、芯片公司或 AI infra 创业公司。

**次选：如果暂时转不了，就在手机 NPU 方向主动争取“设计型任务”。**
比如融合策略设计、图优化 pass、量化方案、runtime profiling、算子性能模型、端云协同部署，而不是只做业务模型适配。

**不建议：为了“更算法”而盲目转普通算法岗。**
除非你能进入真正的大模型训练、后训练、多模态、RL、Agent research 团队，否则很多算法岗的职业杠杆未必比 AI infra 高。

一句话结论：

> **你现在的直觉是对的：云侧 PyTorch/vLLM/训练推理框架适配，比手机 NPU 业务部署更通用、更接近 AI infra 主战场。建议转，但要确保新岗位有框架、算子、runtime、性能设计 ownership，而不是只做适配杂活。**

[1]: https://pytorch.org/blog/executorch-becomes-part-of-pytorch-core/ "ExecuTorch Becomes a Part of PyTorch Core to Expand On-Device Inference Capabilities – PyTorch"
[2]: https://pytorch.org/blog/pytorch-foundation-welcomes-vllm/ "PyTorch Foundation Welcomes vLLM as a Hosted Project – PyTorch"
[3]: https://github.com/vllm-project/vllm-acd "GitHub - vllm-project/vllm-acd: Community maintained hardware plugin for vLLM on acd · GitHub"
[4]: https://www.gartner.com/en/newsroom/press-releases/2026-1-15-gartner-says-worldwide-ai-spending-will-total-2-point-5-trillion-dollars-in-2026 "Gartner Says Worldwide AI Spending Will Total $2.5 Trillion in 2026"
[5]: https://www.gartner.com/en/newsroom/press-releases/2025-10-15-gartner-says-artificial-intelligence-optimized-iaas-is-poised-to-become-the-next-growth-engine-for-artificial-intelligence-infrastructure "Gartner Says AI-Optimized IaaS Is Poised to Become the Next Growth Engine for AI Infrastructure"
[6]: https://file.finance.sina.com.cn/211.154.219.97%3A9494/MRGG/CNSESH_STOCK/2026/2026-4/2026-04-27/12200843.PDF "摩尔线程智能科技（北京）股份有限公司2025年年度报告"
[7]: https://github.com/acd/pytorch "GitHub - acd/pytorch: acd PyTorch adapter (torch_npu). Mirror of https://gitcode.com/acd/pytorch · GitHub"
[8]: https://support.hxwcloud.com/intl/en-us/bestpractice-modelarts/modelarts_llm_infer_5906001.html "Introduction to acd-vLLM_Adapting Mainstream Open-Source Models to acd-vLLM for NPU Inference Based on Lite Server (New)_LLM Inference_Best Practices_ModelArts-hxw Cloud"

谋定而后动