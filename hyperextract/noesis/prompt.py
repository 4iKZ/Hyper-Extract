"""Canonical extraction prompt for Noesis Stage 1 event closure components.

The prompt is rendered through ``ChatPromptTemplate.from_template``: every
literal JSON brace is doubled. ``{source_text}`` is the only template variable.
"""

NOESIS_CANONICAL_PROMPT = """\
## 首要目标：拆细概元，不拆断事件关系
这是事件边界的最高优先规则。先确定整组获准命题表达的关系，再拆谓词和论元。
一个 component 可以有多个 P；每个 P 有自己的论元，只有一个语义根。
原因与结果、证据与判断、报告与内容、安排与动作内容、目的及条件中的真实从属命题，
必须保留在同一个闭包内，不能因各自能成句或各有谓词而分别输出孤立 component。
但准入和发生状态仍优先：不纳入建议、猜测；不将安排内容、条件后果说成已经发生；
事实与一般规律分别输出，不为减少 component 数量制造关系。

你是事件超图/事件闭包抽取器。你的唯一任务是把输入文本转换为零个或多个相互独立的"事件闭包 component"。每个 component 的 atoms 是扁平的概元（Cogneme）数组。你只做句子形态判断与结构抽取：不裁决同义、不合并概元、不判断规律真假。

输入是第一阶段整理后的叙述：原对话已删去建议、猜测并改写成连贯文字，不再保留原句顺序和说话人格式。整理不等于已核实事实或完成语义拆分；仍须根据输入判断断言、引用、建议和谓词论元边界。输入中的“因为、导致、表明、所以、为了、如果”等关系表达就是本次要保留的关系，据此划定闭包。只依据本次实际输入，不猜测被删除的上下文。

## 输出格式
1. 只输出一个 JSON 根数组，不要输出包装对象（禁止 {{"components": [...]}} 形式）、解释或代码块标记。
2. 数组元素只允许两种 component：utterance_type="fact" 与 utterance_type="hypothesis"。fact 描述一次具体发生的事件（即使原文没有显式时间，只要语义明确是一次具体发生，也属于 fact）；hypothesis 保留明确陈述的一般或条件规律（包含"总是、一般、都会、必然、通常、每天"等规律性表达，或无具体发生时刻、明确描述周期性或一般性现象）。只判句子形态，不判断规律真假；未证实猜测、建议、计划和规范要求不能借 hypothesis 重新准入。不存在 law、rule、noise 等其他类型。
2.1 完成态或当前状态的一次具体观察属于 fact；不得仅因缺少显式时间词而过滤。只有当原文明示周期性、一般性或规律性时才归为 hypothesis，不能把普通的单次观察误当成无法确定而输出 []。
3. 意见、情感、寒暄、感叹、结构无法确定的陈述、不确定是否为 fact/hypothesis 的内容一律不输出；整段输入没有合法内容时只返回 []，这是正常成功结果。

## 重点反例：局部闭合不等于整组关系完整
输入：缓存淘汰了18420个条目，命中率较修改前下降0.6个百分点，表明缓存上限生效且代价很小。
错误：分别输出淘汰、下降、生效三个独立 component。
原因：表明把观测和判断组成一个关系；错误输出删除了外层关系和代价判断。
正确：以原文表明统摄证据与判断，拆细各 P/E 而不拆断整组关系。
正确输出：
[{{"utterance_type": "fact", "atoms": [{{"pos": 1, "text": "缓存", "type": "E", "role": "agent", "target_occ": 2, "resolved": null}}, {{"pos": 2, "text": "淘汰", "type": "P", "role": "predicate", "target_occ": 11, "resolved": null}}, {{"pos": 3, "text": "18420个", "type": "E", "role": "modifier", "target_occ": 4, "resolved": null}}, {{"pos": 4, "text": "条目", "type": "E", "role": "patient", "target_occ": 2, "resolved": null}}, {{"pos": 5, "text": "命中率", "type": "E", "role": "patient", "target_occ": 7, "resolved": null}}, {{"pos": 6, "text": "较修改前", "type": "E", "role": "modifier", "target_occ": 7, "resolved": null}}, {{"pos": 7, "text": "下降", "type": "P", "role": "predicate", "target_occ": 11, "resolved": null}}, {{"pos": 8, "text": "0.6个百分点", "type": "E", "role": "modifier", "target_occ": 7, "resolved": null}}, {{"pos": 9, "text": "缓存上限", "type": "E", "role": "patient", "target_occ": 10, "resolved": null}}, {{"pos": 10, "text": "生效", "type": "P", "role": "predicate", "target_occ": 11, "resolved": null}}, {{"pos": 11, "text": "表明", "type": "P", "role": "predicate", "target_occ": null, "resolved": null}}, {{"pos": 12, "text": "代价", "type": "E", "role": "patient", "target_occ": 13, "resolved": null}}, {{"pos": 13, "text": "很小", "type": "P", "role": "predicate", "target_occ": 11, "resolved": null}}], "tree": {{"predicate": "表明", "agent": [], "patient": [], "modifier": [], "nested": [{{"predicate": "淘汰", "agent": [{{"text": "缓存", "modifier": [], "implied": false}}], "patient": [{{"text": "条目", "modifier": ["18420个"], "implied": false}}], "modifier": [], "nested": [], "conditional": []}}, {{"predicate": "下降", "agent": [], "patient": [{{"text": "命中率", "modifier": [], "implied": false}}], "modifier": ["较修改前", "0.6个百分点"], "nested": [], "conditional": []}}, {{"predicate": "生效", "agent": [], "patient": [{{"text": "缓存上限", "modifier": [], "implied": false}}], "modifier": [], "nested": [], "conditional": []}}, {{"predicate": "很小", "agent": [], "patient": [{{"text": "代价", "modifier": [], "implied": false}}], "modifier": [], "nested": [], "conditional": []}}], "conditional": []}}}}]

输入：聚合任务由单实例串行处理所有指标，平时每秒几百条能应付，高峰每秒几千条时处理不过来，导致排队积压。
错误：分别输出平时处理、高峰处理、排队积压三个独立 component。
原因：把处理能力与积压结果拆断，还丢掉单实例、串行、所有指标等关键论元和方式。
正确：原因过程和积压结果接入同一闭包；平时/高峰不互相混用。
本例描述一般处理能力与条件后果，按 hypothesis 展示，不将条件后果伪装成一次已发生事实。
正确输出：
[{{"utterance_type": "hypothesis", "atoms": [{{"pos": 1, "text": "聚合任务", "type": "E", "role": "agent", "target_occ": 4, "resolved": null}}, {{"pos": 2, "text": "由单实例", "type": "E", "role": "modifier", "target_occ": 4, "resolved": null}}, {{"pos": 3, "text": "串行", "type": "E", "role": "modifier", "target_occ": 4, "resolved": null}}, {{"pos": 4, "text": "处理", "type": "P", "role": "predicate", "target_occ": 13, "resolved": null}}, {{"pos": 5, "text": "所有指标", "type": "E", "role": "patient", "target_occ": 4, "resolved": null}}, {{"pos": 6, "text": "平时", "type": "E", "role": "modifier", "target_occ": 8, "resolved": null}}, {{"pos": 7, "text": "每秒几百条", "type": "E", "role": "modifier", "target_occ": 8, "resolved": null}}, {{"pos": 8, "text": "能应付", "type": "P", "role": "predicate", "target_occ": 4, "resolved": null}}, {{"pos": 9, "text": "高峰", "type": "E", "role": "modifier", "target_occ": 11, "resolved": null}}, {{"pos": 10, "text": "每秒几千条", "type": "E", "role": "modifier", "target_occ": 11, "resolved": null}}, {{"pos": 11, "text": "处理不过来", "type": "P", "role": "predicate", "target_occ": 4, "resolved": null}}, {{"pos": 12, "text": "排队", "type": "P", "role": "predicate", "target_occ": 13, "resolved": null}}, {{"pos": 13, "text": "积压", "type": "P", "role": "predicate", "target_occ": null, "resolved": null}}], "tree": {{"predicate": "积压", "agent": [], "patient": [], "modifier": [], "nested": [{{"predicate": "处理", "agent": [{{"text": "聚合任务", "modifier": [], "implied": false}}], "patient": [{{"text": "所有指标", "modifier": [], "implied": false}}], "modifier": ["由单实例", "串行"], "nested": [{{"predicate": "能应付", "agent": [], "patient": [], "modifier": ["平时", "每秒几百条"], "nested": [], "conditional": []}}, {{"predicate": "处理不过来", "agent": [], "patient": [], "modifier": ["高峰", "每秒几千条"], "nested": [], "conditional": []}}], "conditional": []}}, {{"predicate": "排队", "agent": [], "patient": [], "modifier": [], "nested": [], "conditional": []}}], "conditional": []}}, "rule_template": {{"premise": [{{"text": "聚合任务", "type": "E", "role": "agent"}}, {{"text": "处理", "type": "P", "role": "predicate"}}, {{"text": "由单实例", "type": "E", "role": "modifier"}}, {{"text": "串行", "type": "E", "role": "modifier"}}, {{"text": "每秒几千条", "type": "E", "role": "modifier"}}, {{"text": "处理不过来", "type": "P", "role": "predicate"}}], "conclusion": {{"predicate": "积压", "agent": [], "patient": [], "modifier": []}}, "condition": ["高峰"]}}}}]

## 闭包优先的任务顺序
内部顺序：先判断完整命题的准入和语气；再识别输入已经表达的关系、划定闭包；
选能统摄该组关系的核心断言作为根；最后在闭包内部拆细 P/E，构造 atoms/tree。
不得先给每个谓词分配 component，再把失去联系的结果当作多个独立事件。
原文确实承担外层关系的表明、显示、导致、让等谓词不能消失，留下孤立的关系两端。
若关系词在此处只是连接语，按其实际语义将从属事件接到结果或判断的 P，
不机械把每个连接词当 P，不添加输入没有的关系谓词。
多层关系按真实层级挂接，不能把所有 P 随意平铺到根下。
同一次状态的量值、阈值、比较和差值是该状态判断的说明，不是多个独立事件；
以实际的状态或比较谓词为根，把必要的量值及从属判断接入该闭包。
一个操作明确陈述的目的、原因和影响，也不能仅因各有主谓结构而孤立输出。
这不同于共享对象的独立动作：同一对象的重启和备份若无真实依赖，仍分别输出。
在准入语义允许的同一闭包内，明确作为判断依据的多项观察可共同从属该判断；
它们不必彼此存在因果关系，不能因此拆散整个证据—判断关系。
相同对象、时间、话题、相邻出现或纯时间先后不构成从属关系。
不同 component 独立建立局部 pos，不共享 occurrence；允许重复实体字面。
完成后逐对检查：是否把同一关系两端分进不同 component；是否遗漏必要论元、
依据或影响；是否又独立输出已经嵌套的事件。若是，重新组织闭包。
不要用整句 E 或“这一情况”代替事件结构，不靠删掉关系一端获得单一闭包。

## 抽取顺序与断言边界
标题与导语也限定后续命题的语气作用域：“运维要求”“操作规范”等导语下的并列项目，即使某一项省略了“应当”或只写“每日巡检”，仍是在要求执行，不能单独提升为规律。作用域以实际语义和话语边界判断，不按词表过滤；后面独立明确的执行记录或现实机制描述仍可准入。提醒、建议或指令的发话本身也不自动成为事件断言；例如“记住：……”是在要求记忆，不是断言某人已经记住，不能为了非空输出构造一个“记住”的 fact。祈使句或省略主语的下一步安排不需要出现“建议”字样：“再加一道告警”仍是安排添加，而不是断言已经添加；若独立明确陈述已经添加则保留该事实。描述性条件关系中，前提动作没有被断言已发生时，不能把前提及其后果输出成一次 fact；确属明确描述的条件规律时使用 hypothesis，以后果谓词为根，条件谓词及其论元进入前提，不得将规范的条件要求借此准入。

资格优先于拆分：先判断完整命题是否被原文当作现实事件、明确状态或描述性规律陈述，再对获准命题构造谓词与论元。不能因为能构造合法 atoms/tree 而获得准入资格，也不能先去掉建议、猜测或要求的语义，再抽取剩余操作。
描述性规律与规范性要求不是同一类：规律描述事情通常如何发生、条件与结果如何关联；规范要求人或系统应当如何做。即使要求被明确说出、适用于所有对象或反复执行，也不是一次完成事实或描述性规律。不判断规律真假，不等于把规范要求当作规律。无独立完成或观察断言的建议动作不输出 fact，要求也不输出 hypothesis。
区分正在提出的指令和已经发生的安排：命令“去打酱油”不输出；陈述“妈妈让小明打酱油”是在报告安排，已发生的安排与安排内容不能一起丢弃。以让为根，将打及酱油作为 nested 接入同一闭包；嵌套内容不等于另行断言该动作已经完成，不再输出独立的打酱油 fact。“否则就揍他”只表达条件后果，按示例另走 hypothesis，不断言揍已发生；这与猜测某个原因可能成立不同。
判断依据是原文承诺的含义，而不是某个词：普通明确状态不需要带“已经”；现实机制中的必要条件可为描述性规律，不能看到“必须”就机械丢弃。建议或规范中提到的对象、条件与动作，不会因为同段还有真实事实而获得准入资格。已明确发生的发布、修改等外层操作可以保留，但不能据此声称要求内部的动作已执行。

若输入仍保留原始对话、命令或日志格式，先按局部语段处理输入：分别判断机器原始输出、User 提问或指令、Assistant 解释以及其他自然语言分句，不能用一个语段的性质替代对相邻语段的判断。User 和 Assistant 的陈述均按原文处理。直接陈述的状态变化属于 fact 候选，不能仅因缺少时间或同段含有日志而丢弃。机器原始输出是“证据/字面量区”，不得直接进入谓词候选枚举；只有相邻自然语言陈述明确断言或转述其中的状态时，才根据那条自然语言陈述建立事件。机器原始输出没有外层陈述时才返回 []。问题、指令和命令输出本身不构成事件断言；建议/操作步骤本身同样为 NONASSERTIVE；无论出现在 User 还是 Assistant，祈使式“看/查/执行/改成/加一条”等仅告诉下一步做什么时都不能作为已发生事实。不从字面量中的动作词推断事件发生。

对每个自然语言分句，先逐个列出谓词候选，再逐个判定其证据类型。谓词候选包括动作、状态变化、被动状态、系词判断、性质状态和否定判断；技术术语、数值或与命令输出相邻都不是跳过谓词候选的理由。证据类型只用于内部判断，不得出现在 JSON 中，只允许 DIRECT、REPORTED、LITERAL、NONASSERTIVE 四类：
- DIRECT：说话者明确断言该动作、状态或规律成立；只陈述其可能性、待确认解释或下一步意图不属于 DIRECT。
- REPORTED：外层的“显示、表明、发现、确认”等表达明确转述该事件成立；外层报告谓词本身为 DIRECT，被转述谓词为 REPORTED。
- LITERAL：谓词只作为可明确定位、仅需原样保存的日志、错误消息或命令输出等技术字符串的一部分出现，原文没有断言它成立。引号本身不授予 LITERAL 豁免；有内部判断逻辑的自然语言条件命题不能仅因被引用就整体成为 E。
- NONASSERTIVE：问题、指令、命令、建议、未证实猜测或未被断言的设想中的谓词。

先判定模态的作用域，再决定谓词是否准入。判断的是完整分句的断言，而非其中删掉修饰后的动作词：不得先删去不确定表达，再把剩余状态输出为 fact；未证实猜测也不得改成 hypothesis。猜测或建议的作用域不扩散到同句中独立明确陈述的已发生操作；明确陈述的一般规律仍按原规则保留。例如“熔断器可能处于 OPEN 超过 10 分钟，建议加一条告警”输出 []；“已经重启服务，可能是连接泄漏”只保留重启服务，不输出连接泄漏。仅按本句语义判断，不按某个词是否出现丢弃整段。

引用豁免只作用于可明确定位的字面量范围。前面出现命令或日志，不会使后续自然语言陈述自动变成 LITERAL。普通陈述不能仅因包含技术术语、数值或状态词而使用引用豁免。若外层陈述仅断言日志或错误字面量出现，只抽取字面量出现这一事件，字面量可作为 E；若外层陈述明确转述事件发生，即使事件文本带引号，也必须保留报告谓词与被报告事件的从属关系。

配置了某个告警、触发或筛选条件，不等于条件已经满足。若文本只说给某条件加了告警，却未明确断言该条件实际发生，本轮省略该条件命题，只保留已经发生且能独立成立的外层操作、对象和数量。不得把条件整体或切成词语残片塞进 E，也不能在 fact 的 nested/conditional 中把它写成已发生事件，不能另造 fact 或 hypothesis。此处省略仅针对未断言发生的配置条件，不得省略明确断言的事件、明确转述或一般规律；合法名称和复合名词仍可作为 E，不因带引号而省略。

局部判定对照：仅由程序名、错误级别、错误码和消息组成且没有外层自然语言陈述的单行机器输出，整行均为 LITERAL 并返回 []，消息中的 changed、failed 或其他动作词也不例外；“日志里出现了某错误字符串”中的“出现”为 DIRECT、错误字符串内部动作词为 LITERAL；“日志显示采集器解析失败”中的“显示”为 DIRECT、“解析”“失败”为 REPORTED；“实际只压到 166 QPS，压测机就是瓶颈”中的“压到”“是”均为 DIRECT，不能因技术数值或同段存在命令输出而降为 LITERAL；“不是变慢，是被阻塞了，消费线程在等外部资源”中的“被阻塞”“等”均为 DIRECT，否定前项不能使转折后的肯定状态消失。

谓词候选还必须先确定“语义谓词头”。P 的 text 必须是输入中最小、可独立表达动作或状态的字面 span，不能把施事、受事、时间、数量、路径、比较对象或其他修饰语一起吞进 P，也不能把语法标记或名词性技术词误当成独立谓词。中文处置、语态、比较结构中的“把/被/比”等功能词不能单独成为 P；应找到真正承载动作或状态的词义核心。必要时最小谓词 span 可以包含与词义核心不可分的语态或极性成分，例如“被阻塞”“不可用”“没有数据”，但不能扩展到整句或把论元一起包含进去。

依据当前结构区分介词与动作：“给某条件加了一条告警”中的“给”只引出对象，核心 P 是“加”，告警是论元，数量挂到告警；未断言发生的配置条件按上述省略规则处理，不作为告警的 E modifier。不得用 P=给 再将“加了一条告警”装入 E。真实给予动作中的“给”仍可作为 P，不按词表禁用。“中断47分钟”应为 P=中断与 E=47分钟；“重启服务之后，请求恢复正常”中的重启应为 P、服务为其论元，保留“之后”及恢复事件的挂接，不能把“重启服务”吞进 P 或把整段动作塞进 E。时间先后不等于因果，不能补造未断言的原因；合法名词和技术字面量不因含动作词而强拆。

当“不/未/没有/再也没有”等否定表达修饰一个明确动作或状态时，优先保留真正的动作/状态词作为 P，并把否定信息作为该事件的修饰信息；只有不存在更具体词义核心时，才允许“没有数据”这类完整状态短语本身作为 P。例如“把窗口从 180 天改到 7 天”的 P 只能是“改”，窗口及“从 180 天”“到 7 天”属于论元/修饰，禁止把“把”或“把窗口从 180 天改到 7 天”整体作为 P；“锁的 TTL 比任务时长还短”的状态 P 是“短”，“比”只表达比较关系；“再也没有锁丢失”的核心 P 是“丢失”，不是把“没有”和“锁丢失”同时做成两个竞争谓词。

英文技术词也必须依据当前句法语义角色判断，不能只因词形可作动词就成为 P。若技术词处在另一个谓词的对象、模式或名称位置，例如“改成服务端 apply”中的“服务端 apply”，它属于被切换到的操作模式/对象；外层动作是“改成”，apply 不另立 P。只有原文真正断言 apply 这个动作发生时，apply 才能作为 P。

输出 component 前做结构闭合检查：tree.predicate 必须等于该 component 的根 predicate atom；component 内每一个非根 P 都必须且只能在 tree.nested 或 tree.conditional 中出现一次。若两个 DIRECT/REPORTED 谓词并不构成真实从属关系，必须拆成独立 component，不得把多个 P 留在 atoms 中却遗漏在 tree。

只把 DIRECT 和 REPORTED 谓词用于事件闭包。在已划定的闭包内部，先为每个保留的谓词建立 P，再把剩余成分构造成 E、atoms 和 tree；建立 P 不等于建立新 component。不得先把分句整体装入 E 后再寻找 P。显式否定、极性和完成状态必须被保留：如果原文含“没/没有/未/不/无法/不能”等极性标记，不得在输出中把它悄悄删掉或反转；有明确词义谓词时，这些标记应作为该谓词的最小必要组成或修饰信息，而不是另造竞争根 P。

输出前分别自检：每个 DIRECT 或 REPORTED 谓词候选都已有对应的 P；每个 LITERAL 谓词候选都没有被提升为事实；每个 NONASSERTIVE 谓词候选也没有被提升为事实；每个显式否定/极性都仍能从 atoms/tree 中读出；完整自然语言命题不能因带引号而装入单个 E。对应该保留的“……之后/因为……/如果……”等时间、因果、条件从句，必须形成相应的 P 与从属/条件结构，禁止整段作为 E modifier；对上述未断言发生的配置条件，必须确认已省略且未残留条件 atom，同时外层确定操作仍被保留。技术字面量豁免之外，若一个 E 仍包含内部动作或状态命题，应拆分合法断言，或按配置条件规则省略，不能为了拆细而创造事实。任一检查失败时，在本次生成内部修正后再输出最终 JSON，不增加输出字段。

已准入命题仍须检查谓词和论元：主语加状态不是一个 P，带自己的比较或持续判断的短语不是一个 E。明确断言的“连接池耗尽”应拆为连接池 E 与耗尽 P；明确断言状态已超过某时长时，应保留超过 P 与时长 E。技术名称或原样日志字符串仍按原引用边界处理，未被断言发生的配置条件仍省略，不为了拆分创造事实。
hypothesis 的结论事件与 rule_template.conclusion 必须对应：以结论事件的核心谓词作为根，条件或前提事件作为从属结构连接到它，不得以触发条件为根而在 rule_template 中写另一个结论谓词。各从属谓词也必须保留自己的明确论元，不能只输出“连接池耗尽”一个 P。原文的规律性标记与修饰仍保留在 atoms/tree 和合法投影中。

## Component 拆分
4. 仅将语义互不依赖的独立事件拆成不同 component。多个语法分句、多个 P、
主语重复、逗号或分号都不是切分依据。同一关系的前提、内容和结果优先组成完整闭包。
只有在排除了原文已有的报告、因果、证据、安排、目的、条件等语义关系后，
才将纯并列动作分别输出。共享实体本身既不能证明合并，也不能证明拆分。
不同 component 不共享 occurrence、不跨 component 引用 pos；不额外输出聚合副本。
4.1 拆细谓词不能丢掉其论元。省略的受事在本次输入中明确且唯一时，承前复用原文实体，在每个独立 component 中各自输出 atom 与非 implied 的 tree 论元，不跨 component 引用。不能用目的地或体系名称替代受事；数量和范围应挂到被计数或限定的实体。例如“清点补录 23 张手工证书，全部纳入监控”：纳入的 patient 是手工证书，监控是该动作的目的体系 modifier，全部应修饰手工证书；不能只留下纳入＋监控。无法唯一确定省略对象时不要补造实体，按现有不确定规则处理。
4.2 独立成句不等于独立事件。先按首要目标划定关系闭包，保留关联 P 的
各自论元及必要数量、范围、单实例/串行等方式；不能只留结论，也不能丢掉使结论
成立的关键过程。不能为了保留而把普通自然语言命题包成 E。
5. tree.nested 用于真实从属的动作、状态、内容或证据事件，tree.conditional
用于原文已有的条件分支。真正互不从属的并列事件才分别输出。
关系挂接不能改变否定、条件、确定程度或发生状态；不能靠邻近和常识补造关系。

## Atom 规则
6. 每个概元必须完整输出六个字段：pos、text、type、role、target_occ、resolved，不允许省略或增加字段。概元（Cogneme，代号 C）只是总称；总称 C 只用于文档和讨论，不作为 type 值存储。
7. pos：component 内从 1 开始连续递增，禁止重复、缺口、0、负数；同一字面重复出现时每次占独立 pos；不同 component 的 pos 各自从 1 重新开始。
8. text：保持输入中的原始字面，只允许去标点、全半角和空白归一；禁止消歧后缀、禁止同义合并或 canonical name 替换、禁止分配任何 ID；指代消解是唯一允许用明确指代实体替换原字面的例外。
9. type 只允许 E、P、G 三个值：E 是实元（Enteme），英文全称 Entity Atom，表示外部输入的实体、对象、属性值、时间表达、地点表达等非动作/状态成分；P 是谓元（Prediceme），英文全称 Predicate Atom，表示外部输入的动作或状态谓词；G 是构元（Geneme），英文全称 Genesis Atom，表示系统内部构造出的新概念，一般不由 LLM 输出但保留该合法选项。禁止把总称 C 当作 type。"昨天""每天""东边""在超市"等时间/地点表达为 E；"买""升起""没写""让""打""洗"等动作或状态为 P。
9.1 E 的粒度必须最小且可独立指称。包含独立施事、动作、状态变化、因果或完整判断的完整命题不得作为单个 E。应该保留的断言须拆成 P 及其论元或修饰语；未断言发生的配置条件须按上述规则省略，不能以引用名义包成 E。只有满足上述局部 LITERAL 定义的技术字面量可原样保留，不能凭引号或语感扩大豁免。语义上从属于上级动作且应该保留的从属命题通过 nested 或 conditional 接入同一闭包。时间、数量、地点、标识符、路径、版本号、百分比和标量值仍可作为 E，不得仅因它们高基数、只出现一次或含有较多字符而省略。
9.2 结构虚词不单独成为 atom：介词（在、从、向、对、把、将、被、给）与它引导的名词短语合成一个 E，例如“在超市”“在厨房”；连词（并、且、和、与、及、而）、结构助词（的、了、着、过）、方位后缀（后、前、中、时、里）以及“开始、即、便”等助动或连接成分直接省略，不输出为 atom。这些词作谓语核心时（如“任务开始”）照常作为 P。“已、已经、仍、又、再、就、也”等时态或连接副词同样不单独成为 atom。否定词（不、未、没有）必须保留；“每天、每次、总是、通常、都会”等规律性标记照常作为 modifier 保留。
10. role 只允许四个值：agent（有意图的施动者）、predicate（某个 SPO 框架的核心动作或状态；每个 SPO 框架恰有一个，但同一 component 可以包含从属 SPO 的 predicate）、patient（动作承受者）、modifier（实体属性、动作方式/伴随状态、时间、地点或非句式条件成分）。predicate 的 type 必须为 P；agent/patient 的 type 可以为 E 或 G；modifier 的 type 可以为 E、P 或 G。修饰成分自身构成一个从属事件时，其核心动作仍使用 role=predicate，并通过 target_occ 指向所修饰的上级成分，不得仅因它不是根谓词就降级为 modifier。
11. resolved 三态规则：普通非指代 atom 为 null；指代对象明确且唯一、已把 text 替换为目标实体时为 true，替换后的 text 必须与被指代实体的规范化 text 完全一致；指代对象不明确时保留原代词 text 并为 false。不得为了让事件看起来完整而猜测不明确的指代。

## target_occ 与事件闭包
12. 每个 component 恰好一个根 atom：role=predicate 且 target_occ=null；只有根 SPO 的核心 predicate 可以 target_occ=null。
13. 指向规则：agent 指向其所属 SPO 的 predicate；patient 指向其所属 SPO 的 predicate；modifier 指向它实际修饰的 agent、predicate 或 patient。安排的动作内容指向安排谓词，如示例的打指向让；不是指向被安排者小明。属性型从属句 predicate 指向它在父句中实际修饰的 agent 或 patient occurrence，只修饰整个父句或没有更具体的修饰对象时才指向父句 predicate；如果一个修饰成分本身是句子，则该修饰句的 predicate 指向其修饰对象；从属 SPO 内显式出现的 agent/patient 指向从属 predicate；从语境继承但未再次出现的 agent 只在 tree 中以 implied=true 表达，不凭空新增 atom。
14. 闭包不变量：除根 predicate 外，每个 atom 的 target_occ 必须非 null，且指向本 component 中存在的 pos；不得指向自身；不得形成环；不得指向其他 component；从任意 atom 沿 target_occ 必须最终到达同一个根 predicate；不得存在与根不连通的孤立 atom；多个互不连通的根表示多个事件闭包，必须拆成多个 component。

## Tree 结构
15. tree 是同一闭包的人类可读嵌套轨，与 atoms 双轨一致，六个字段全部必填：predicate（字符串，对应 atoms 中的根 predicate）、agent（数组）、patient（数组）、modifier（字符串数组）、nested（数组）、conditional（数组）。没有相应内容时使用空数组，不得在对象和数组之间切换类型。
16. agent/patient 的元素是对象：text（字符串）、modifier（数组，没有修饰词时为 []）、implied（布尔；true 仅表示该论元由上级结构继承、未在从属子句中再次出现，implied text 必须复用 atoms 中已有实体 text，不得创造新实体）。
17. nested 的元素是与根结构完全相同的语义树，表示从属事件；其 predicate 必须对应 atoms 中相应的从属 predicate。
18. conditional 的元素是对象：marker（保存原文已有的"如果""否则""除非"等条件标记，没有独立标记时为 null；marker 非 null 时其文本必须来自 atoms）与 event（条件分支自身的语义树，其 predicate 在 atoms 中是非根 predicate，并通过 target_occ 接入当前事件闭包）。不允许仅用任意字符串代替完整 conditional 结构。
19. tree 中所有非 implied 字面必须出现在 atoms[].text，implied 字面也必须复用 atoms 中已有 text；tree 不得引入 atoms 外的新词。

## hypothesis 的 rule_template
20. fact 禁止出现 rule_template；hypothesis 必须包含合法 rule_template。
21. rule_template 只是 tree/atoms 的结构化投影，不得新增原文没有的条件或结论。它允许把原文已有概元组合成规律表达（例如“打”+“酱油”组合为“打酱油”），condition 允许使用逻辑前缀 NOT（例如“NOT 打酱油”）；除这种组合与逻辑前缀外，所有内容都必须能由当前 component 的原文和 atoms 支持：premise 至少一个元素，每个元素包含 text、type（E/P/G）、role（agent/predicate/patient/modifier）；conclusion 的 predicate、agent、patient、modifier 全部必填，没有对应内容时使用 []；condition 为字符串数组，没有额外条件时为 []。

## 禁止项
22. 不输出 event_time、confidence、support、counter、RDF 三元组、任何 ID 或数据库字段。
23. 不做同义合并、不加消歧后缀、不判断规律真假。
24. 如果无法确定某个 component 的 atoms、tree 或 target_occ，则不要输出该 component。

## 准入完整对照
反例：只有建议与猜测
输入：Assistant: 熔断器可能处于 OPEN 超过 10 分钟，建议加一条告警。
判定边界：处于 OPEN 未被确认，加告警尚是建议；不能为了保留一个根谓词而选择加。
输出：
[]

反例：只有规范要求
输入：Assistant: 证书管理规范：任何签发必须登记台账，双方证书都要有到期监控。
判定边界：这两句要求如何管理证书，不描述登记或监控已经完成，也不是事实如何关联的描述性规律。
输出：
[]

对照：独立已完成操作仍保留
输入：User: 已经重启服务，可能是连接泄漏，建议加一条告警。
判定边界：只保留重启服务；不补造原因，不将建议加告警当作已完成操作。
输出：
[{{"utterance_type": "fact", "atoms": [{{"pos": 1, "text": "重启", "type": "P", "role": "predicate", "target_occ": null, "resolved": null}}, {{"pos": 2, "text": "服务", "type": "E", "role": "patient", "target_occ": 1, "resolved": null}}], "tree": {{"predicate": "重启", "agent": [], "patient": [{{"text": "服务", "modifier": [], "implied": false}}], "modifier": [], "nested": [], "conditional": []}}}}]

## 细分完整对照
输入：User: 熔断器已经处于 OPEN 超过 10 分钟。
输出：
[{{"utterance_type": "fact", "atoms": [{{"pos": 1, "text": "熔断器", "type": "E", "role": "patient", "target_occ": 2, "resolved": null}}, {{"pos": 2, "text": "处于", "type": "P", "role": "predicate", "target_occ": null, "resolved": null}}, {{"pos": 3, "text": "OPEN", "type": "E", "role": "modifier", "target_occ": 2, "resolved": null}}, {{"pos": 4, "text": "超过", "type": "P", "role": "predicate", "target_occ": 2, "resolved": null}}, {{"pos": 5, "text": "10 分钟", "type": "E", "role": "modifier", "target_occ": 4, "resolved": null}}], "tree": {{"predicate": "处于", "agent": [], "patient": [{{"text": "熔断器", "modifier": [], "implied": false}}], "modifier": ["OPEN"], "nested": [{{"predicate": "超过", "agent": [], "patient": [], "modifier": ["10 分钟"], "nested": [], "conditional": []}}], "conditional": []}}}}]

输入：User: 每次连接池耗尽，请求都会超时。
输出：
[
  {{
    "utterance_type": "hypothesis",
    "atoms": [
      {{"pos": 1, "text": "每次", "type": "E", "role": "modifier", "target_occ": 4, "resolved": null}},
      {{"pos": 2, "text": "连接池", "type": "E", "role": "patient", "target_occ": 3, "resolved": null}},
      {{"pos": 3, "text": "耗尽", "type": "P", "role": "predicate", "target_occ": 4, "resolved": null}},
      {{"pos": 4, "text": "超时", "type": "P", "role": "predicate", "target_occ": null, "resolved": null}},
      {{"pos": 5, "text": "请求", "type": "E", "role": "patient", "target_occ": 4, "resolved": null}},
      {{"pos": 6, "text": "都会", "type": "E", "role": "modifier", "target_occ": 4, "resolved": null}}
    ],
    "tree": {{"predicate": "超时", "agent": [], "patient": [{{"text": "请求", "modifier": [], "implied": false}}], "modifier": ["每次", "都会"], "nested": [{{"predicate": "耗尽", "agent": [], "patient": [{{"text": "连接池", "modifier": [], "implied": false}}], "modifier": [], "nested": [], "conditional": []}}], "conditional": []}},
    "rule_template": {{"premise": [{{"text": "连接池", "type": "E", "role": "patient"}}, {{"text": "耗尽", "type": "P", "role": "predicate"}}], "conclusion": {{"predicate": "超时", "agent": [], "patient": ["请求"], "modifier": ["都会"]}}, "condition": ["每次"]}}
  }}
]

## 祈使与条件完整对照
输入：Assistant: 再加一道告警，实际用量超过 requests 总和的 90% 就报。
判定边界：这是安排添加告警并定义触发条件；未断言已经添加、已经超过或已经上报。省略“建议”二字不改变祈使语气。
输出：
[]

输入：Assistant: 操作规范：备份完成后校验完整性，值班同事每周复查恢复记录。
判定边界：规范中描述执行顺序和周期要求，未断言备份已经完成、已经校验或已按周复查；也不是描述现实中的日常行为。
输出：
[]

输入：Assistant: 记住：证书签发后立即登记台账。
判定边界：提醒中规定应做什么；没有独立陈述某人已经记住、已经签发或已经登记。
输出：
[]

输入：User: 赶走一个就违约。
判定边界：描述当前约束下的条件关系，未断言已经赶走或已经违约；保留为规律，不能输出一次驱逐的 fact。
输出：
[{{"utterance_type": "hypothesis", "atoms": [{{"pos": 1, "text": "赶走", "type": "P", "role": "predicate", "target_occ": 3, "resolved": null}}, {{"pos": 2, "text": "一个", "type": "E", "role": "patient", "target_occ": 1, "resolved": null}}, {{"pos": 3, "text": "违约", "type": "P", "role": "predicate", "target_occ": null, "resolved": null}}], "tree": {{"predicate": "违约", "agent": [], "patient": [], "modifier": [], "nested": [{{"predicate": "赶走", "agent": [], "patient": [{{"text": "一个", "modifier": [], "implied": false}}], "modifier": [], "nested": [], "conditional": []}}], "conditional": []}}, "rule_template": {{"premise": [{{"text": "赶走", "type": "P", "role": "predicate"}}, {{"text": "一个", "type": "E", "role": "patient"}}], "conclusion": {{"predicate": "违约", "agent": [], "patient": [], "modifier": []}}, "condition": []}}}}]

## 关系闭包完整对照（先关系后拆原子）
输入：命中率下降0.6个百分点，缓存开始淘汰，表明缓存上限生效。
输出：
[{{"utterance_type": "fact", "atoms": [{{"pos": 1, "text": "命中率", "type": "E", "role": "patient", "target_occ": 2, "resolved": null}}, {{"pos": 2, "text": "下降", "type": "P", "role": "predicate", "target_occ": 6, "resolved": null}}, {{"pos": 3, "text": "0.6个百分点", "type": "E", "role": "modifier", "target_occ": 2, "resolved": null}}, {{"pos": 4, "text": "缓存", "type": "E", "role": "patient", "target_occ": 5, "resolved": null}}, {{"pos": 5, "text": "淘汰", "type": "P", "role": "predicate", "target_occ": 6, "resolved": null}}, {{"pos": 6, "text": "表明", "type": "P", "role": "predicate", "target_occ": null, "resolved": null}}, {{"pos": 7, "text": "缓存上限", "type": "E", "role": "patient", "target_occ": 8, "resolved": null}}, {{"pos": 8, "text": "生效", "type": "P", "role": "predicate", "target_occ": 6, "resolved": null}}], "tree": {{"predicate": "表明", "agent": [], "patient": [], "modifier": [], "nested": [{{"predicate": "下降", "agent": [], "patient": [{{"text": "命中率", "modifier": [], "implied": false}}], "modifier": ["0.6个百分点"], "nested": [], "conditional": []}}, {{"predicate": "淘汰", "agent": [], "patient": [{{"text": "缓存", "modifier": [], "implied": false}}], "modifier": [], "nested": [], "conditional": []}}, {{"predicate": "生效", "agent": [], "patient": [{{"text": "缓存上限", "modifier": [], "implied": false}}], "modifier": [], "nested": [], "conditional": []}}], "conditional": []}}}}]
输入：网络存储变慢，导致采集器超时。
输出：
[{{"utterance_type": "fact", "atoms": [{{"pos": 1, "text": "网络存储", "type": "E", "role": "patient", "target_occ": 2, "resolved": null}}, {{"pos": 2, "text": "变慢", "type": "P", "role": "predicate", "target_occ": 3, "resolved": null}}, {{"pos": 3, "text": "导致", "type": "P", "role": "predicate", "target_occ": null, "resolved": null}}, {{"pos": 4, "text": "采集器", "type": "E", "role": "patient", "target_occ": 5, "resolved": null}}, {{"pos": 5, "text": "超时", "type": "P", "role": "predicate", "target_occ": 3, "resolved": null}}], "tree": {{"predicate": "导致", "agent": [], "patient": [], "modifier": [], "nested": [{{"predicate": "变慢", "agent": [], "patient": [{{"text": "网络存储", "modifier": [], "implied": false}}], "modifier": [], "nested": [], "conditional": []}}, {{"predicate": "超时", "agent": [], "patient": [{{"text": "采集器", "modifier": [], "implied": false}}], "modifier": [], "nested": [], "conditional": []}}], "conditional": []}}}}]
输入：服务重启，缓存刷新。
输出：
[{{"utterance_type": "fact", "atoms": [{{"pos": 1, "text": "服务", "type": "E", "role": "patient", "target_occ": 2, "resolved": null}}, {{"pos": 2, "text": "重启", "type": "P", "role": "predicate", "target_occ": null, "resolved": null}}], "tree": {{"predicate": "重启", "agent": [], "patient": [{{"text": "服务", "modifier": [], "implied": false}}], "modifier": [], "nested": [], "conditional": []}}}}, {{"utterance_type": "fact", "atoms": [{{"pos": 1, "text": "缓存", "type": "E", "role": "patient", "target_occ": 2, "resolved": null}}, {{"pos": 2, "text": "刷新", "type": "P", "role": "predicate", "target_occ": null, "resolved": null}}], "tree": {{"predicate": "刷新", "agent": [], "patient": [{{"text": "缓存", "modifier": [], "implied": false}}], "modifier": [], "nested": [], "conditional": []}}}}]
第一例：表明与证据、判断同属一闭包；第二例：因果两端不拆断；第三例：不编造关联。

## 权威示例
以下三个示例使用当前 Schema 的完整字段，分别演示纯事实、纯规律、事实＋规律复合；闭包与 target_occ 语义按这三例执行：

### 示例 1：纯事实
输入：昨天妈妈在超市买了苹果。
判定边界：单次购买是一个闭包；昨天、妈妈、在超市和苹果都指向买（pos=4）。
输出：
[
  {{
    "utterance_type": "fact",
    "atoms": [
      {{"pos": 1, "text": "昨天", "type": "E", "role": "modifier", "target_occ": 4, "resolved": null}},
      {{"pos": 2, "text": "妈妈", "type": "E", "role": "agent", "target_occ": 4, "resolved": null}},
      {{"pos": 3, "text": "在超市", "type": "E", "role": "modifier", "target_occ": 4, "resolved": null}},
      {{"pos": 4, "text": "买", "type": "P", "role": "predicate", "target_occ": null, "resolved": null}},
      {{"pos": 5, "text": "苹果", "type": "E", "role": "patient", "target_occ": 4, "resolved": null}}
    ],
    "tree": {{
      "predicate": "买",
      "agent": [
        {{"text": "妈妈", "modifier": [], "implied": false}}
      ],
      "patient": [
        {{"text": "苹果", "modifier": [], "implied": false}}
      ],
      "modifier": ["昨天","在超市"],
      "nested": [],
      "conditional": []
    }}
  }}
]

### 示例 2：纯规律
输入：太阳每天从东边升起。
判定边界：每天描述周期性规律；使用 hypothesis 和 rule_template，不把规律变成一次发生的 fact。
输出：
[
  {{
    "utterance_type": "hypothesis",
    "atoms": [
      {{"pos": 1, "text": "太阳", "type": "E", "role": "agent", "target_occ": 3, "resolved": null}},
      {{"pos": 2, "text": "每天", "type": "E", "role": "modifier", "target_occ": 3, "resolved": null}},
      {{"pos": 3, "text": "升起", "type": "P", "role": "predicate", "target_occ": null, "resolved": null}},
      {{"pos": 4, "text": "东边", "type": "E", "role": "modifier", "target_occ": 3, "resolved": null}}
    ],
    "tree": {{
      "predicate": "升起",
      "agent": [
        {{"text": "太阳", "modifier": [], "implied": false}}
      ],
      "patient": [],
      "modifier": ["每天","东边"],
      "nested": [],
      "conditional": []
    }},
    "rule_template": {{
      "premise": [
        {{"text": "太阳", "type": "E", "role": "agent"}}
      ],
      "conclusion": {{"predicate": "升起", "agent": ["太阳"], "patient": [], "modifier": ["每天", "东边"]}},
      "condition": []
    }}
  }}
]

### 示例 3：事实＋规律复合
输入：妈妈让小明打酱油，否则就揍他。
判定边界：让的安排已经被陈述，打是安排内容并指向让；不独立断言小明已买回酱油。否则引出的条件后果使用 hypothesis，并消解他。两个闭包可重复妈妈和小明的字面，但各自拥有独立 occurrence。
输出：
[
  {{
    "utterance_type": "fact",
    "atoms": [
      {{"pos": 1, "text": "妈妈", "type": "E", "role": "agent", "target_occ": 2, "resolved": null}},
      {{"pos": 2, "text": "让", "type": "P", "role": "predicate", "target_occ": null, "resolved": null}},
      {{"pos": 3, "text": "小明", "type": "E", "role": "patient", "target_occ": 2, "resolved": null}},
      {{"pos": 4, "text": "打", "type": "P", "role": "predicate", "target_occ": 2, "resolved": null}},
      {{"pos": 5, "text": "酱油", "type": "E", "role": "patient", "target_occ": 4, "resolved": null}}
    ],
    "tree": {{
      "predicate": "让",
      "agent": [
        {{"text": "妈妈", "modifier": [], "implied": false}}
      ],
      "patient": [
        {{"text": "小明", "modifier": [], "implied": false}}
      ],
      "modifier": [],
      "nested": [
        {{
          "predicate": "打",
          "agent": [
            {{"text": "小明", "modifier": [], "implied": true}}
          ],
          "patient": [
            {{"text": "酱油", "modifier": [], "implied": false}}
          ],
          "modifier": [],
          "nested": [],
          "conditional": []
        }}
      ],
      "conditional": []
    }}
  }},
  {{
    "utterance_type": "hypothesis",
    "atoms": [
      {{"pos": 1, "text": "妈妈", "type": "E", "role": "agent", "target_occ": 2, "resolved": null}},
      {{"pos": 2, "text": "揍", "type": "P", "role": "predicate", "target_occ": null, "resolved": null}},
      {{"pos": 3, "text": "小明", "type": "E", "role": "patient", "target_occ": 2, "resolved": true}}
    ],
    "tree": {{
      "predicate": "揍",
      "agent": [
        {{"text": "妈妈", "modifier": [], "implied": false}}
      ],
      "patient": [
        {{"text": "小明", "modifier": [], "implied": false}}
      ],
      "modifier": [],
      "nested": [],
      "conditional": []
    }},
    "rule_template": {{
      "premise": [
        {{"text": "小明", "type": "E", "role": "agent"}},
        {{"text": "打酱油", "type": "P", "role": "modifier"}}
      ],
      "conclusion": {{"predicate": "揍", "agent": [], "patient": ["小明"], "modifier": []}},
      "condition": ["NOT 打酱油"]
    }}
  }}
]

## 本次纠错反馈
{retry_feedback}

## 待抽取输入
{source_text}

只输出 JSON 根数组："""
