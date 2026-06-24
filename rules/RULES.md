# CustomRule YAML 编写参考

这个文件给 AI/人类阅读，用来生成真实 rule。真实可执行规则请写在 `rules/*.yaml` 或 `rules/*.yml` 中。

一个 YAML 文件可以包含多个 rule：

```yaml
rules:
  - name: 规则名称
    enabled: true
    activity:
      - 当前页面 Activity 全名
    conditions:
      task_contains: 可选，任务文本必须包含的字符串或字符串列表
      task_not_contains: 可选，任务文本不能包含的字符串或字符串列表
      app_name: 可选，当前 app 名称，字符串或字符串列表
      step: 可选，仅在第 N 个 agent step 触发
    actions:
      - type: action 类型
        # action 参数
    step_delay: 1.0
    max_fires: 1
    post_delay: 3.0
    terminal: false
    context_note: >
      规则执行后注入给模型的提示。
```

`activity` 是第一层性能筛选条件，必须写成列表。rule 数量变多时，系统会先用当前 Activity 找候选 rule，再检查 `conditions`。

## 可选 Actions

### `tap_element`

按 UI 元素选择器点击控件。

可用选择器：

- `text`：精确匹配控件文字
- `textContains`：匹配包含某段文字的控件
- `textStartsWith`：匹配文字前缀
- `resource_id`：Android resource-id
- `content_desc`：无障碍描述
- `class_name`：控件 className
- `timeout`：等待元素出现的秒数，默认 5

示例：

```yaml
- type: tap_element
  text: 登录
```

### `type_into_element`

按 UI 元素选择器找到输入框并输入文本。

必填：

- `input_text`：要输入的文本

可用选择器同 `tap_element`。

可选：

- `clear`：是否先清空，默认 true
- `timeout`：等待元素出现的秒数，默认 5

示例：

```yaml
- type: type_into_element
  input_text: "15311699022"
  text: 手机号码
```

### `play_audio_while_holding_element`

长按手机端元素，同时在电脑端播放音频。

必填：

- `audio_path`：音频文件路径，相对路径默认从项目根目录查找

可用选择器同 `tap_element`，用来定位需要长按的按钮。

可选：

- `min_hold_seconds`：最小长按秒数，默认 3.0
- `press_lead_seconds`：先按住多久后开始播放音频，默认 0.3
- `hold_seconds`：固定长按秒数；不填时使用音频时长 + 1 秒
- `timeout`：等待元素出现的秒数，默认 5
- `player`：播放器命令，macOS 默认 `afplay`

示例：

```yaml
- type: play_audio_while_holding_element
  audio_path: xiaoxiao.wav
  textContains: 说中文
  timeout: 5
  min_hold_seconds: 3.0
  press_lead_seconds: 0.3
```

### `do`

直接构造普通模型动作。

常用参数：

- `action`: `Tap` / `Type` / `Swipe` / `Back` / `Home` / `Wait` / `Note` / `Take_over`
- `element` / `start` / `end` / `text` / `duration` / `message`

示例：

```yaml
- type: do
  action: Tap
  element: [500, 800]
```

### `finish`

结束任务。

常用参数：

- `message`：结束信息

示例：

```yaml
- type: finish
  message: 示例任务已完成。
```

## 示例规则

### 登录

```yaml
rules:
  - name: 超级app-登录
    enabled: true
    activity:
      - com.wakeup.howear.login.LoginActivity
    actions:
      - type: type_into_element
        input_text: "15311699022"
        text: 手机号码
      - type: type_into_element
        input_text: "meng1234"
        text: 请输入密码
      - type: tap_element
        resource_id: com.wakeup.howear:id/agreeCheckBox
      - type: tap_element
        text: 登录
    step_delay: 1.0
    max_fires: 1
    post_delay: 3.0
    terminal: false
    context_note: >
      【系统提示】已自动完成登录。
      无需再执行登录操作，继续完成原任务。
```

### 音频播放并长按

```yaml
rules:
  - name: 音频类-播放音频并长按语音按钮
    enabled: true
    activity:
      - com.wakeup.feature.translate.TranslateHomeActivity
    conditions:
      task_contains: audio
    actions:
      - type: play_audio_while_holding_element
        audio_path: xiaoxiao.wav
        textContains: 说中文
        timeout: 5
        min_hold_seconds: 3.0
        press_lead_seconds: 0.3
    max_fires: 1
    post_delay: 3.0
    terminal: false
    context_note: >
      【系统提示】已执行音频类自动规则：
      电脑端已播放测试音频，手机端已长按语音按钮完成一次语音输入。
      请根据当前截图继续检查识别结果是否符合测试用例预期。
```
