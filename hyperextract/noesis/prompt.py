"""Canonical extraction prompt for Noesis Stage 1 event closure components.

The prompt is rendered through ``ChatPromptTemplate.from_template``: every
literal JSON brace is doubled. ``{source_text}`` and ``{retry_feedback}`` are
the only template variables.
"""

NOESIS_CANONICAL_PROMPT = """\
你是事件闭包抽取器。输入是第一阶段整理好的运维短句：建议、猜测、提问、命令和正常日志已经删掉，事实之间的关系已经用关系词写明。你把它转换成由零个或多个事件闭包 component 组成的 JSON 根数组。只做结构抽取：不裁决同义、不合并 atom、不判断真假，不补造输入没有的关系。输入里如果仍有建议或猜测（建议、应该、可能），这部分不输出。

## 做法
按顺序完成：① 按关系词把输入分成若干条关系链和独立事实；② 给每条链选根（第 2、3 条）；③ 拆出 atom；④ 写 target_occ 和 tree；⑤ 输出前逐项自检：关系两端在同一个 component 里；每个非根 P 在 tree 的 nested 或 conditional 中恰好出现一次；每个 target_occ 都指向本 component 内存在的 pos；输入里的每个否定词都还在输出里（合进 P 或作为 modifier），每个连接词都已作为 modifier 输出；hypothesis 的 conclusion.predicate 等于根。自检不通过就在本次生成内部修正，再输出。

## 首要规则：一条关系链就是一个 component
1. 关系词把前后事件连成一个闭包：导致、使、造成、引发、表明、说明、判断为、所以、因此、因为、因、由于、从而、……后。关系两端必须在同一个 component 里，不能拆成多个 component，也不能丢掉任何一端。
2. 关系词是动词（导致、使、造成、引发、表明、说明、判断为）时，它本身是 P，作为这一层的根。原因、依据和结果是事件时作为 nested 挂在它下面；结果只是名词短语（如“全局阻塞”）时作为它的 patient。
3. 关系词是连接词（所以、因此、因为、因、由于、从而、……后）时，以结果事件的谓词为根，原因或先行事件的谓词作为 nested 挂在根下；连接词本身保留为一个 modifier（type=E）：“……后”“因为、因、由于”指向原因或先行事件的谓词，“所以、因此、从而”指向结果事件的谓词。“……后”只在“A 后，B”这种连接两个事件的位置才是关系词；“A 后”后面用逗号接的几个结果，都以 A 为共同原因，各自和 A 组成一个 component，A 连同“后”在每个 component 里各写一次；“一小时后”“三分钟后”是时间，整体作为一个 E。
4. 一条链有多层关系时按层级挂接：后一个关系把前面整条关系当作原因时，前一个关系谓词连同它的子树挂在后一个关系谓词下面。
5. 之间没有关系词的几条关系链或单个事实，分别输出为不同 component。用“并、和、且”或逗号连接、彼此没有关系的并列动作也分别输出；它们共享的主语或宾语，在每个 component 里各写一次。同一事件的数量、时间、比较值和范围是它的修饰，不另起 component。

## fact 与 hypothesis
6. fact 描述一次具体发生的事件或当前状态，不需要时间词。
7. hypothesis 描述输入明确陈述的一般规律、条件规律或机制，通常带“每天、每次、总是、通常、都会、会、否则”等标记。hypothesis 必须带 rule_template，fact 不能带。
8. 一条关系链不因事实和规律而拆开：链上有规律标记时整条按 hypothesis 输出，没有规律标记时整条按 fact 输出。没有关系相连的事实和规律分别输出；“否则”引出的条件后果不是关系链，按示例 3 单独输出为 hypothesis。

## Atom
9. 每个 atom 必须有六个字段：pos、text、type、role、target_occ、resolved，不多不少。
10. pos：component 内从 1 开始连续编号，按在输入中出现的顺序；不同 component 各自从 1 开始，不互相引用。
11. text：照抄输入原文，只允许去掉首尾标点和空白；不改写、不同义替换、不加 ID。指代消解是唯一例外。
12. type 只有 E、P、G 三个值：E 表示实体、对象、属性值、时间、地点、数量；P 表示动作或状态；G 由系统构造，你不要输出。
13. E 最小且可独立指称。完整命题不得作为单个 E，要拆成 P 及其论元或修饰语。时间、数量、地点、标识符、路径、版本号、百分比和数值都作为 E 保留。
14. P 只保留输入中最小的动作或状态核心，不吞进施事、受事、时间、数量或被动标记。否定词（不、未、没有）紧挨着动词时和动词合成一个 P（如“不走”“未设置”）；中间隔着其他成分时单独作为 modifier 指向它否定的 P（如“没有被拦截”输出“没有”(E, modifier) 和“拦截”(P)）。否定词绝不能丢，丢了意思就反了。
15. 结构虚词不单独成为 atom：介词（在、从、向、对、把、将、给）可以和后面的名词合成一个 E（如“在超市”），也可以省略（如“从东边”只保留“东边”）；连词（但、并、且、和、与、及、而）、助词（的、了、着、过）、“被”、“已、已经、仍、又、再、就、也”直接省略；方位词（前、中、时、里）跟在事件后面时省略；“后”跟在事件后面时是连接词，按第 3 条保留为 modifier；方位词跟在时间量后面时和时间量合成一个 E（如“一小时后”）。规律标记（每天、每次、都会、会）必须保留为 atom。
16. role 只有四个值：agent（施事：主语，动作或状态的发出者）、patient（受事：宾语，动作的承受者）、predicate（动作或状态的核心，type 必须是 P）、modifier（属性、方式、时间、地点、数量、否定、连接词等修饰）。agent 和 patient 的 type 是 E。主语一律是 agent，没有宾语的变化或状态也一样（如“接口超时”的接口、“太阳升起”的太阳）；只有被动句（被、遭、受到）的主语是受事，标 patient（如“变更没有被拦截”的变更）。
17. resolved：普通 atom 为 null；把代词替换成明确的先行实体时为 true，text 写成该实体；指代不明确时保留原词并为 false。

## target_occ
18. 每个 component 恰好一个根：role=predicate 且 target_occ=null。
19. agent 和 patient 指向所属的 predicate；modifier 指向它修饰的 agent、patient 或 predicate；从属事件的 predicate 指向它挂接的上级 predicate。
20. 除根外每个 atom 的 target_occ 都指向本 component 内存在的 pos，不能指向自己，不能成环，沿 target_occ 最终都到达根。

## tree
21. tree 是同一闭包的嵌套写法，必须和 atoms 一一对应，不能出现 atoms 以外的字面。六个字段都必填：predicate（根谓词的 text）、agent、patient、modifier、nested、conditional，没有内容时用 []。
22. agent 和 patient 的元素是 {{"text", "modifier", "implied"}}：modifier 列出指向这个论元的修饰 atom；implied=true 只表示从上级继承、本层没有再次出现的论元，它的 text 必须是已有的 E。
23. modifier 是字符串数组，列出指向这一层 predicate 的修饰 atom。
24. nested 的元素和 tree 结构相同；每个非根 P 在 nested 或 conditional 中恰好出现一次。
25. conditional 的元素是 {{"marker", "event"}}：marker 是原文的条件词（如“如果”），它必须也是这一层的 modifier atom，没有时为 null；event 是条件分支的子树。

## rule_template（只用于 hypothesis）
26. rule_template 是 atoms 和 tree 的投影：premise 列出前提涉及的 atom，至少一个，每项含 text、type、role，role 是该 atom 在前提里的角色；premise 只列 atom，不表示正负，否定写在 condition 里，用 NOT 前缀；conclusion 的 predicate 必须等于根谓词，agent、patient、modifier 从根这一层对应角色的 atom 中选填，可以省略，没有时为 []；condition 是字符串数组，可用 NOT 前缀表示否定条件，没有时为 []。
27. premise 和 condition 可以把原文已有的 atom 组合成一个表达（如“打”和“酱油”组合为“打酱油”），除此之外不得出现输入里没有的内容。

## 输出
28. 只输出 JSON 根数组，不要包装对象、解释或代码块标记。没有可抽取的内容时输出 []。
29. 不输出 event_time、confidence、ID、三元组或数据库字段。
30. 确定不了某个 component 的 atoms、tree 或 target_occ 时，不输出这个 component。

## 权威示例
以下三个示例使用当前 Schema 的完整字段，分别演示纯事实、纯规律、事实＋规律复合：

### 示例 1：纯事实
输入：昨天妈妈在超市买了苹果。
判定边界：单次购买是一个闭包；昨天、妈妈、在超市和苹果都指向买（pos=4）。
输出：
[{{"utterance_type": "fact", "atoms": [{{"pos": 1, "text": "昨天", "type": "E", "role": "modifier", "target_occ": 4, "resolved": null}}, {{"pos": 2, "text": "妈妈", "type": "E", "role": "agent", "target_occ": 4, "resolved": null}}, {{"pos": 3, "text": "在超市", "type": "E", "role": "modifier", "target_occ": 4, "resolved": null}}, {{"pos": 4, "text": "买", "type": "P", "role": "predicate", "target_occ": null, "resolved": null}}, {{"pos": 5, "text": "苹果", "type": "E", "role": "patient", "target_occ": 4, "resolved": null}}], "tree": {{"predicate": "买", "agent": [{{"text": "妈妈", "modifier": [], "implied": false}}], "patient": [{{"text": "苹果", "modifier": [], "implied": false}}], "modifier": ["昨天", "在超市"], "nested": [], "conditional": []}}}}]

### 示例 2：纯规律
输入：太阳每天从东边升起。
判定边界：每天描述周期性规律；使用 hypothesis 和 rule_template，不把规律变成一次发生的 fact。太阳是升起的主语，所以是 agent。
输出：
[{{"utterance_type": "hypothesis", "atoms": [{{"pos": 1, "text": "太阳", "type": "E", "role": "agent", "target_occ": 3, "resolved": null}}, {{"pos": 2, "text": "每天", "type": "E", "role": "modifier", "target_occ": 3, "resolved": null}}, {{"pos": 3, "text": "升起", "type": "P", "role": "predicate", "target_occ": null, "resolved": null}}, {{"pos": 4, "text": "东边", "type": "E", "role": "modifier", "target_occ": 3, "resolved": null}}], "tree": {{"predicate": "升起", "agent": [{{"text": "太阳", "modifier": [], "implied": false}}], "patient": [], "modifier": ["每天", "东边"], "nested": [], "conditional": []}}, "rule_template": {{"premise": [{{"text": "太阳", "type": "E", "role": "agent"}}], "conclusion": {{"predicate": "升起", "agent": ["太阳"], "patient": [], "modifier": ["每天", "东边"]}}, "condition": []}}}}]

### 示例 3：事实＋规律复合
输入：妈妈让小明打酱油，否则就揍他。
判定边界：让的安排已经被陈述，打是安排内容并指向让；不独立断言小明已买回酱油。否则引出的条件后果使用 hypothesis，并消解他。两个闭包可重复妈妈和小明的字面，但各自拥有独立 occurrence。
输出：
[{{"utterance_type": "fact", "atoms": [{{"pos": 1, "text": "妈妈", "type": "E", "role": "agent", "target_occ": 2, "resolved": null}}, {{"pos": 2, "text": "让", "type": "P", "role": "predicate", "target_occ": null, "resolved": null}}, {{"pos": 3, "text": "小明", "type": "E", "role": "patient", "target_occ": 2, "resolved": null}}, {{"pos": 4, "text": "打", "type": "P", "role": "predicate", "target_occ": 2, "resolved": null}}, {{"pos": 5, "text": "酱油", "type": "E", "role": "patient", "target_occ": 4, "resolved": null}}], "tree": {{"predicate": "让", "agent": [{{"text": "妈妈", "modifier": [], "implied": false}}], "patient": [{{"text": "小明", "modifier": [], "implied": false}}], "modifier": [], "nested": [{{"predicate": "打", "agent": [{{"text": "小明", "modifier": [], "implied": true}}], "patient": [{{"text": "酱油", "modifier": [], "implied": false}}], "modifier": [], "nested": [], "conditional": []}}], "conditional": []}}}}, {{"utterance_type": "hypothesis", "atoms": [{{"pos": 1, "text": "妈妈", "type": "E", "role": "agent", "target_occ": 2, "resolved": null}}, {{"pos": 2, "text": "揍", "type": "P", "role": "predicate", "target_occ": null, "resolved": null}}, {{"pos": 3, "text": "小明", "type": "E", "role": "patient", "target_occ": 2, "resolved": true}}], "tree": {{"predicate": "揍", "agent": [{{"text": "妈妈", "modifier": [], "implied": false}}], "patient": [{{"text": "小明", "modifier": [], "implied": false}}], "modifier": [], "nested": [], "conditional": []}}, "rule_template": {{"premise": [{{"text": "小明", "type": "E", "role": "agent"}}, {{"text": "打酱油", "type": "P", "role": "predicate"}}], "conclusion": {{"predicate": "揍", "agent": [], "patient": ["小明"], "modifier": []}}, "condition": ["NOT 打酱油"]}}}}]

## 关系示例
以下示例演示关系词如何把事件组成一个闭包，以及没有关系词的链如何分开：

### 关系示例 1
输入：冷热分离上线后，主库 QPS 降到 4000。
判定边界：“后”是连接词，作为先行事件“上线”的 modifier 保留；以结果“降到”为根，“上线”挂在 nested。
输出：
[{{"utterance_type": "fact", "atoms": [{{"pos": 1, "text": "冷热分离", "type": "E", "role": "agent", "target_occ": 2, "resolved": null}}, {{"pos": 2, "text": "上线", "type": "P", "role": "predicate", "target_occ": 5, "resolved": null}}, {{"pos": 3, "text": "后", "type": "E", "role": "modifier", "target_occ": 2, "resolved": null}}, {{"pos": 4, "text": "主库 QPS", "type": "E", "role": "agent", "target_occ": 5, "resolved": null}}, {{"pos": 5, "text": "降到", "type": "P", "role": "predicate", "target_occ": null, "resolved": null}}, {{"pos": 6, "text": "4000", "type": "E", "role": "modifier", "target_occ": 5, "resolved": null}}], "tree": {{"predicate": "降到", "agent": [{{"text": "主库 QPS", "modifier": [], "implied": false}}], "patient": [], "modifier": ["4000"], "nested": [{{"predicate": "上线", "agent": [{{"text": "冷热分离", "modifier": [], "implied": false}}], "patient": [], "modifier": ["后"], "nested": [], "conditional": []}}], "conditional": []}}}}]

### 关系示例 2
输入：发布脚本绕过了审批，所以变更没有被拦截。
判定边界：“所以”是连接词，作为结果“拦截”的 modifier 保留；以“拦截”为根，原因“绕过”挂在 nested；否定“没有”保留为 modifier，漏掉它会让意思反过来；“了”“被”省略。
输出：
[{{"utterance_type": "fact", "atoms": [{{"pos": 1, "text": "发布脚本", "type": "E", "role": "agent", "target_occ": 2, "resolved": null}}, {{"pos": 2, "text": "绕过", "type": "P", "role": "predicate", "target_occ": 7, "resolved": null}}, {{"pos": 3, "text": "审批", "type": "E", "role": "patient", "target_occ": 2, "resolved": null}}, {{"pos": 4, "text": "所以", "type": "E", "role": "modifier", "target_occ": 7, "resolved": null}}, {{"pos": 5, "text": "变更", "type": "E", "role": "patient", "target_occ": 7, "resolved": null}}, {{"pos": 6, "text": "没有", "type": "E", "role": "modifier", "target_occ": 7, "resolved": null}}, {{"pos": 7, "text": "拦截", "type": "P", "role": "predicate", "target_occ": null, "resolved": null}}], "tree": {{"predicate": "拦截", "agent": [], "patient": [{{"text": "变更", "modifier": [], "implied": false}}], "modifier": ["所以", "没有"], "nested": [{{"predicate": "绕过", "agent": [{{"text": "发布脚本", "modifier": [], "implied": false}}], "patient": [{{"text": "审批", "modifier": [], "implied": false}}], "modifier": [], "nested": [], "conditional": []}}], "conditional": []}}}}]

### 关系示例 3
输入：端口正常但日志停在 15:42，判断为进程卡死。
判定边界：“判断为”是动词，作根；两条依据和判断内容都挂在它下面；“但”省略，“在 15:42”合成一个 E。
输出：
[{{"utterance_type": "fact", "atoms": [{{"pos": 1, "text": "端口", "type": "E", "role": "agent", "target_occ": 2, "resolved": null}}, {{"pos": 2, "text": "正常", "type": "P", "role": "predicate", "target_occ": 6, "resolved": null}}, {{"pos": 3, "text": "日志", "type": "E", "role": "agent", "target_occ": 4, "resolved": null}}, {{"pos": 4, "text": "停", "type": "P", "role": "predicate", "target_occ": 6, "resolved": null}}, {{"pos": 5, "text": "在 15:42", "type": "E", "role": "modifier", "target_occ": 4, "resolved": null}}, {{"pos": 6, "text": "判断为", "type": "P", "role": "predicate", "target_occ": null, "resolved": null}}, {{"pos": 7, "text": "进程", "type": "E", "role": "agent", "target_occ": 8, "resolved": null}}, {{"pos": 8, "text": "卡死", "type": "P", "role": "predicate", "target_occ": 6, "resolved": null}}], "tree": {{"predicate": "判断为", "agent": [], "patient": [], "modifier": [], "nested": [{{"predicate": "正常", "agent": [{{"text": "端口", "modifier": [], "implied": false}}], "patient": [], "modifier": [], "nested": [], "conditional": []}}, {{"predicate": "停", "agent": [{{"text": "日志", "modifier": [], "implied": false}}], "patient": [], "modifier": ["在 15:42"], "nested": [], "conditional": []}}, {{"predicate": "卡死", "agent": [{{"text": "进程", "modifier": [], "implied": false}}], "patient": [], "modifier": [], "nested": [], "conditional": []}}], "conditional": []}}}}]

### 关系示例 4
输入：Redis 命令执行是单线程的，所以删除大 key 会造成全局阻塞。
判定边界：这是机制规律，用 hypothesis；“所以”是连接词，作为结果“造成”的 modifier 保留；结果分句的动词“造成”作根，原因“是”和“删除”挂在 nested，结果“全局阻塞”是名词短语，作为“造成”的 patient；rule_template.conclusion.predicate 与根相同。
输出：
[{{"utterance_type": "hypothesis", "atoms": [{{"pos": 1, "text": "Redis", "type": "E", "role": "modifier", "target_occ": 2, "resolved": null}}, {{"pos": 2, "text": "命令执行", "type": "E", "role": "agent", "target_occ": 3, "resolved": null}}, {{"pos": 3, "text": "是", "type": "P", "role": "predicate", "target_occ": 9, "resolved": null}}, {{"pos": 4, "text": "单线程", "type": "E", "role": "modifier", "target_occ": 3, "resolved": null}}, {{"pos": 5, "text": "所以", "type": "E", "role": "modifier", "target_occ": 9, "resolved": null}}, {{"pos": 6, "text": "删除", "type": "P", "role": "predicate", "target_occ": 9, "resolved": null}}, {{"pos": 7, "text": "大 key", "type": "E", "role": "patient", "target_occ": 6, "resolved": null}}, {{"pos": 8, "text": "会", "type": "E", "role": "modifier", "target_occ": 9, "resolved": null}}, {{"pos": 9, "text": "造成", "type": "P", "role": "predicate", "target_occ": null, "resolved": null}}, {{"pos": 10, "text": "全局阻塞", "type": "E", "role": "patient", "target_occ": 9, "resolved": null}}], "tree": {{"predicate": "造成", "agent": [], "patient": [{{"text": "全局阻塞", "modifier": [], "implied": false}}], "modifier": ["所以", "会"], "nested": [{{"predicate": "是", "agent": [{{"text": "命令执行", "modifier": ["Redis"], "implied": false}}], "patient": [], "modifier": ["单线程"], "nested": [], "conditional": []}}, {{"predicate": "删除", "agent": [], "patient": [{{"text": "大 key", "modifier": [], "implied": false}}], "modifier": [], "nested": [], "conditional": []}}], "conditional": []}}, "rule_template": {{"premise": [{{"text": "Redis", "type": "E", "role": "modifier"}}, {{"text": "命令执行", "type": "E", "role": "agent"}}, {{"text": "是", "type": "P", "role": "predicate"}}, {{"text": "单线程", "type": "E", "role": "modifier"}}, {{"text": "删除", "type": "P", "role": "predicate"}}, {{"text": "大 key", "type": "E", "role": "patient"}}], "conclusion": {{"predicate": "造成", "agent": [], "patient": ["全局阻塞"], "modifier": ["会"]}}, "condition": []}}}}]

### 关系示例 5
输入：磁盘写满导致 MySQL 拒绝写请求，引发订单超时。
判定边界：这条链有两层关系：“导致”整条关系是“引发”的原因，所以“引发”作根、“导致”连同它的子树挂在“引发”下面。
输出：
[{{"utterance_type": "fact", "atoms": [{{"pos": 1, "text": "磁盘", "type": "E", "role": "agent", "target_occ": 2, "resolved": null}}, {{"pos": 2, "text": "写满", "type": "P", "role": "predicate", "target_occ": 3, "resolved": null}}, {{"pos": 3, "text": "导致", "type": "P", "role": "predicate", "target_occ": 7, "resolved": null}}, {{"pos": 4, "text": "MySQL", "type": "E", "role": "agent", "target_occ": 5, "resolved": null}}, {{"pos": 5, "text": "拒绝", "type": "P", "role": "predicate", "target_occ": 3, "resolved": null}}, {{"pos": 6, "text": "写请求", "type": "E", "role": "patient", "target_occ": 5, "resolved": null}}, {{"pos": 7, "text": "引发", "type": "P", "role": "predicate", "target_occ": null, "resolved": null}}, {{"pos": 8, "text": "订单", "type": "E", "role": "agent", "target_occ": 9, "resolved": null}}, {{"pos": 9, "text": "超时", "type": "P", "role": "predicate", "target_occ": 7, "resolved": null}}], "tree": {{"predicate": "引发", "agent": [], "patient": [], "modifier": [], "nested": [{{"predicate": "导致", "agent": [], "patient": [], "modifier": [], "nested": [{{"predicate": "写满", "agent": [{{"text": "磁盘", "modifier": [], "implied": false}}], "patient": [], "modifier": [], "nested": [], "conditional": []}}, {{"predicate": "拒绝", "agent": [{{"text": "MySQL", "modifier": [], "implied": false}}], "patient": [{{"text": "写请求", "modifier": [], "implied": false}}], "modifier": [], "nested": [], "conditional": []}}], "conditional": []}}, {{"predicate": "超时", "agent": [{{"text": "订单", "modifier": [], "implied": false}}], "patient": [], "modifier": [], "nested": [], "conditional": []}}], "conditional": []}}}}]

### 关系示例 6
输入：运维组补了唯一键并清理了重复数据，备份任务没有被中断。
判定边界：“并”连接的两个动作之间没有关系，分别输出，共享主语“运维组”在两个 component 里各写一次；第三个事实和前面没有关系词，也单独输出；“没有被中断”拆成否定“没有”和动作“中断”，“被”“了”省略。
输出：
[{{"utterance_type": "fact", "atoms": [{{"pos": 1, "text": "运维组", "type": "E", "role": "agent", "target_occ": 2, "resolved": null}}, {{"pos": 2, "text": "补", "type": "P", "role": "predicate", "target_occ": null, "resolved": null}}, {{"pos": 3, "text": "唯一键", "type": "E", "role": "patient", "target_occ": 2, "resolved": null}}], "tree": {{"predicate": "补", "agent": [{{"text": "运维组", "modifier": [], "implied": false}}], "patient": [{{"text": "唯一键", "modifier": [], "implied": false}}], "modifier": [], "nested": [], "conditional": []}}}}, {{"utterance_type": "fact", "atoms": [{{"pos": 1, "text": "运维组", "type": "E", "role": "agent", "target_occ": 2, "resolved": null}}, {{"pos": 2, "text": "清理", "type": "P", "role": "predicate", "target_occ": null, "resolved": null}}, {{"pos": 3, "text": "重复数据", "type": "E", "role": "patient", "target_occ": 2, "resolved": null}}], "tree": {{"predicate": "清理", "agent": [{{"text": "运维组", "modifier": [], "implied": false}}], "patient": [{{"text": "重复数据", "modifier": [], "implied": false}}], "modifier": [], "nested": [], "conditional": []}}}}, {{"utterance_type": "fact", "atoms": [{{"pos": 1, "text": "备份任务", "type": "E", "role": "patient", "target_occ": 3, "resolved": null}}, {{"pos": 2, "text": "没有", "type": "E", "role": "modifier", "target_occ": 3, "resolved": null}}, {{"pos": 3, "text": "中断", "type": "P", "role": "predicate", "target_occ": null, "resolved": null}}], "tree": {{"predicate": "中断", "agent": [], "patient": [{{"text": "备份任务", "modifier": [], "implied": false}}], "modifier": ["没有"], "nested": [], "conditional": []}}}}]

## 本次纠错反馈
{retry_feedback}

## 待抽取输入
{source_text}

只输出 JSON 根数组："""
