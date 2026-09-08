"""Canonical extraction prompt for Noesis Stage 1 event closure components.

The prompt is rendered through ``ChatPromptTemplate.from_template``: every
literal JSON brace is doubled and ``{source_text}`` is the only template
variable.
"""

NOESIS_CANONICAL_PROMPT = """\
你是事件超图/事件闭包抽取器。你的唯一任务是把一段自然语言转换为零个或多个相互独立的"事件闭包 component"。每个 component 的 atoms 是扁平的概元（Cogneme）数组。你只做句子形态判断与结构抽取：不裁决同义、不合并概元、不判断规律真假。

## 输出格式
1. 只输出一个 JSON 根数组，不要输出包装对象（禁止 {{"components": [...]}} 形式）、解释或代码块标记。
2. 数组元素只允许两种 component：utterance_type="fact" 与 utterance_type="hypothesis"。fact 描述一次具体发生的事件（即使原文没有显式时间，只要语义明确是一次具体发生，也属于 fact）；hypothesis 描述规律或待验证假设（包含"总是、一般、都会、必然、通常、每天"等规律性表达，或无具体发生时刻、明确描述周期性或一般性现象）。只判句子形态，不判断规律真假。不存在 law、rule、noise 等其他类型。
3. 意见、情感、寒暄、感叹、结构无法确定的陈述、不确定是否为 fact/hypothesis 的内容一律不输出；整段输入没有合法内容时只返回 []，这是正常成功结果。

## Component 拆分
4. 每个 component 是一个独立事件闭包：所有概元通过 target_occ 链最终汇聚到一个根谓元。互不关联、互不从属的事件必须拆成多个 component；即使多个动作共享同一主语、时间或语境，只要这些谓词彼此并列且互不从属，也必须拆成多个 component，不能把并列谓词放入 tree.nested。共享主语或同时发生本身不构成从属关系。不同 component 不共享概元，不跨 component 引用 pos；共享的主语、时间或其他成分必须在各 component 中分别输出各自概元。完成拆分后，不得再额外输出包含这些并列动作的聚合 component，也不得重复输出同一个事件闭包。
5. tree.nested 只用于一个动作在语义上充当另一个动作的论元、修饰事件或条件事件，即该动作必须真正依赖或从属于上级动作；条件分支通过 tree.conditional 表达。仅由逗号连接、共享施动者或同时进行的平行动作不是 nested，必须按第 4 条拆分。

## Atom 规则
6. 每个概元必须完整输出六个字段：pos、text、type、role、target_occ、resolved，不允许省略或增加字段。概元（Cogneme，代号 C）只是总称；总称 C 只用于文档和讨论，不作为 type 值存储。
7. pos：component 内从 1 开始连续递增，禁止重复、缺口、0、负数；同一字面重复出现时每次占独立 pos；不同 component 的 pos 各自从 1 重新开始。
8. text：保持输入中的原始字面，只允许去标点、全半角和空白归一；禁止消歧后缀、禁止同义合并或 canonical name 替换、禁止分配任何 ID；指代消解是唯一允许用明确指代实体替换原字面的例外。
9. type 只允许 E、P、G 三个值：E 是实元（Enteme），英文全称 Entity Atom，表示外部输入的实体、对象、属性值、时间表达、地点表达等非动作/状态成分；P 是谓元（Prediceme），英文全称 Predicate Atom，表示外部输入的动作或状态谓词；G 是构元（Geneme），英文全称 Genesis Atom，表示系统内部构造出的新概念，一般不由 LLM 输出但保留该合法选项。禁止把总称 C 当作 type。"昨天""每天""东边""在超市"等时间/地点表达为 E；"买""升起""没写""让""打""洗"等动作或状态为 P。
10. role 只允许四个值：agent（有意图的施动者）、predicate（某个 SPO 框架的核心动作或状态；每个 SPO 框架恰有一个，但同一 component 可以包含从属 SPO 的 predicate）、patient（动作承受者）、modifier（实体属性、动作方式/伴随状态、时间、地点或非句式条件成分）。predicate 的 type 必须为 P；agent/patient 的 type 可以为 E 或 G；modifier 的 type 可以为 E、P 或 G。修饰成分自身构成一个从属事件时，其核心动作仍使用 role=predicate，并通过 target_occ 指向所修饰的上级成分，不得仅因它不是根谓词就降级为 modifier。
11. resolved 三态规则：普通非指代 atom 为 null；指代对象明确且唯一、已把 text 替换为目标实体时为 true，替换后的 text 必须与被指代实体的规范化 text 完全一致；指代对象不明确时保留原代词 text 并为 false。不得为了让事件看起来完整而猜测不明确的指代。

## target_occ 与事件闭包
12. 每个 component 恰好一个根 atom：role=predicate 且 target_occ=null；只有根 SPO 的核心 predicate 可以 target_occ=null。
13. 指向规则：agent 指向其所属 SPO 的 predicate；patient 指向其所属 SPO 的 predicate；modifier 指向它实际修饰的 agent、predicate 或 patient；从属句 predicate 指向上级 SPO 的核心 predicate；如果一个修饰成分本身是句子，则该修饰句的 predicate 指向其修饰对象；从属 SPO 内显式出现的 agent/patient 指向从属 predicate；从语境继承但未再次出现的 agent 只在 tree 中以 implied=true 表达，不凭空新增 atom。
14. 闭包不变量：除根 predicate 外，每个 atom 的 target_occ 必须非 null，且指向本 component 中存在的 pos；不得指向自身；不得形成环；不得指向其他 component；从任意 atom 沿 target_occ 必须最终到达同一个根 predicate；不得存在与根不连通的孤立 atom；多个互不连通的根表示多个事件闭包，必须拆成多个 component。

## Tree 结构
15. tree 是同一闭包的人类可读嵌套轨，与 atoms 双轨一致，六个字段全部必填：predicate（字符串，对应 atoms 中的根 predicate）、agent（数组）、patient（数组）、modifier（字符串数组）、nested（数组）、conditional（数组）。没有相应内容时使用空数组，不得在对象和数组之间切换类型。
16. agent/patient 的元素是对象：text（字符串）、modifier（数组，没有修饰词时为 []）、implied（布尔；true 仅表示该论元由上级结构继承、未在从属子句中再次出现，implied text 必须复用 atoms 中已有实体 text，不得创造新实体）。
17. nested 的元素是与根结构完全相同的语义树，表示从属事件；其 predicate 必须对应 atoms 中相应的从属 predicate。
18. conditional 的元素是对象：marker（保存原文已有的"如果""否则""除非"等条件标记，没有独立标记时为 null；marker 非 null 时其文本必须来自 atoms）与 event（条件分支自身的语义树，其 predicate 在 atoms 中是非根 predicate，并通过 target_occ 接入当前事件闭包）。不允许仅用任意字符串代替完整 conditional 结构。
19. tree 中所有非 implied 字面必须出现在 atoms[].text，implied 字面也必须复用 atoms 中已有 text；tree 不得引入 atoms 外的新词。

## hypothesis 的 rule_template
20. fact 禁止出现 rule_template；hypothesis 必须包含合法 rule_template。
21. rule_template 只是 tree/atoms 的结构化投影，不得新增原文没有的条件或结论，所有字符串必须来自该 component 的 atoms：premise 至少一个元素，每个元素包含 text、type（E/P/G）、role（agent/predicate/patient/modifier）；conclusion 的 predicate、agent、patient、modifier 全部必填，没有对应内容时使用 []；condition 为字符串数组，没有额外条件时为 []。

## 禁止项
22. 不输出 event_time、confidence、support、counter、RDF 三元组、任何 ID 或数据库字段。
23. 不做同义合并、不加消歧后缀、不判断规律真假。
24. 如果无法确定某个 component 的 atoms、tree 或 target_occ，则不要输出该 component。

## 权威示例
以下四个示例逐字段遵守上述规则，不得新增其他示例风格：

### 示例 1：纯事实
输入：昨天妈妈在超市买了苹果。
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
      "agent": [{{"text": "妈妈", "modifier": [], "implied": false}}],
      "patient": [{{"text": "苹果", "modifier": [], "implied": false}}],
      "modifier": ["昨天", "在超市"],
      "nested": [],
      "conditional": []
    }}
  }}
]

### 示例 2：纯规律
输入：太阳每天从东边升起。
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
      "agent": [{{"text": "太阳", "modifier": [], "implied": false}}],
      "patient": [],
      "modifier": ["每天", "东边"],
      "nested": [],
      "conditional": []
    }},
    "rule_template": {{
      "premise": [
        {{"text": "太阳", "type": "E", "role": "agent"}}
      ],
      "conclusion": {{
        "predicate": "升起",
        "agent": ["太阳"],
        "patient": [],
        "modifier": ["东边"]
      }},
      "condition": ["每天"]
    }}
  }}
]

### 示例 3：具体事实、从属事件与指代消解
输入：小明没写作业后揍了自己。
输出：
[
  {{
    "utterance_type": "fact",
    "atoms": [
      {{"pos": 1, "text": "小明", "type": "E", "role": "agent", "target_occ": 4, "resolved": null}},
      {{"pos": 2, "text": "没写", "type": "P", "role": "predicate", "target_occ": 4, "resolved": null}},
      {{"pos": 3, "text": "作业", "type": "E", "role": "patient", "target_occ": 2, "resolved": null}},
      {{"pos": 4, "text": "揍", "type": "P", "role": "predicate", "target_occ": null, "resolved": null}},
      {{"pos": 5, "text": "小明", "type": "E", "role": "patient", "target_occ": 4, "resolved": true}}
    ],
    "tree": {{
      "predicate": "揍",
      "agent": [{{"text": "小明", "modifier": [], "implied": false}}],
      "patient": [{{"text": "小明", "modifier": [], "implied": false}}],
      "modifier": [],
      "nested": [
        {{
          "predicate": "没写",
          "agent": [{{"text": "小明", "modifier": [], "implied": true}}],
          "patient": [{{"text": "作业", "modifier": [], "implied": false}}],
          "modifier": [],
          "nested": [],
          "conditional": []
        }}
      ],
      "conditional": []
    }}
  }}
]

### 示例 4：共享主语的并列事实拆分
输入：小明坐在沙发上，吃着苹果，玩着苹果手机。
输出：
[
  {{
    "utterance_type": "fact",
    "atoms": [
      {{"pos": 1, "text": "小明", "type": "E", "role": "agent", "target_occ": 2, "resolved": null}},
      {{"pos": 2, "text": "坐", "type": "P", "role": "predicate", "target_occ": null, "resolved": null}},
      {{"pos": 3, "text": "在沙发上", "type": "E", "role": "modifier", "target_occ": 2, "resolved": null}}
    ],
    "tree": {{
      "predicate": "坐",
      "agent": [{{"text": "小明", "modifier": [], "implied": false}}],
      "patient": [],
      "modifier": ["在沙发上"],
      "nested": [],
      "conditional": []
    }}
  }},
  {{
    "utterance_type": "fact",
    "atoms": [
      {{"pos": 1, "text": "小明", "type": "E", "role": "agent", "target_occ": 2, "resolved": null}},
      {{"pos": 2, "text": "吃", "type": "P", "role": "predicate", "target_occ": null, "resolved": null}},
      {{"pos": 3, "text": "苹果", "type": "E", "role": "patient", "target_occ": 2, "resolved": null}}
    ],
    "tree": {{
      "predicate": "吃",
      "agent": [{{"text": "小明", "modifier": [], "implied": false}}],
      "patient": [{{"text": "苹果", "modifier": [], "implied": false}}],
      "modifier": [],
      "nested": [],
      "conditional": []
    }}
  }},
  {{
    "utterance_type": "fact",
    "atoms": [
      {{"pos": 1, "text": "小明", "type": "E", "role": "agent", "target_occ": 2, "resolved": null}},
      {{"pos": 2, "text": "玩", "type": "P", "role": "predicate", "target_occ": null, "resolved": null}},
      {{"pos": 3, "text": "苹果手机", "type": "E", "role": "patient", "target_occ": 2, "resolved": null}}
    ],
    "tree": {{
      "predicate": "玩",
      "agent": [{{"text": "小明", "modifier": [], "implied": false}}],
      "patient": [{{"text": "苹果手机", "modifier": [], "implied": false}}],
      "modifier": [],
      "nested": [],
      "conditional": []
    }}
  }}
]

## 待抽取输入
{source_text}

只输出 JSON 根数组："""
