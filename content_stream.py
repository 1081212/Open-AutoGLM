#!/usr/bin/env python3
"""
Simple Content Agent with GLM-4.7-flash (Streaming Version)
Supports calling local phoneagent service at localhost:8000/run
"""

import os
import json
import requests
from typing import List, Dict, Any


class ContentAgentStream:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        self.phoneagent_url = "http://localhost:8000/run"
        self.messages: List[Dict[str, Any]] = []

        # Define the phoneagent tool
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "call_phoneagent",
                    "description": "调用手机代理执行操作，比如打电话、发短信、查看联系人等手机相关操作",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "要执行的操作描述，例如：'给张三打电话'、'发短信给李四说你好'"
                            }
                        },
                        "required": ["prompt"]
                    }
                }
            }
        ]

    def call_phoneagent(self, prompt: str) -> str:
        """Call the local phoneagent service with streaming"""
        print(f"\n🤖 [调用 PhoneAgent] 参数: {prompt}")
        print("📱 [PhoneAgent 执行过程]")
        print("-" * 60)

        try:
            response = requests.post(
                self.phoneagent_url + "/stream",
                json={"prompt": prompt},
                stream=True,
                timeout=300  # 5 minutes timeout
            )
            response.raise_for_status()

            result = None
            success = False

            # Process SSE stream
            for line in response.iter_lines():
                if not line:
                    continue

                line = line.decode('utf-8')
                if not line.startswith('data: '):
                    continue

                data_str = line[6:]  # Remove 'data: ' prefix
                try:
                    data = json.loads(data_str)
                    event_type = data.get('type')

                    if event_type == 'start':
                        print(f"▶️  开始执行任务: {data.get('prompt')}")
                    elif event_type == 'queued':
                        message = data.get('message', '').strip()
                        if message:
                            print(f"   {message}")
                    elif event_type == 'step_start':
                        print(f"\n🔄 Step {data.get('step')} 开始")
                    elif event_type == 'step':
                        step = data.get('step')
                        thinking = data.get('thinking', '').strip()
                        action = data.get('action')
                        message = data.get('message')

                        if thinking:
                            print(f"💭 Step {step} 思考: {thinking}")
                        if action:
                            print(f"🎯 Step {step} 动作: {json.dumps(action, ensure_ascii=False)}")
                        if message:
                            print(f"📝 Step {step} 消息: {message}")
                    elif event_type == 'done':
                        result = data.get('result')
                        success = data.get('success', True)
                        print("-" * 60)
                        print(f"✅ [PhoneAgent 完成] {result}\n")
                    elif event_type == 'error':
                        error_msg = data.get('message')
                        print("-" * 60)
                        print(f"❌ [PhoneAgent 错误] {error_msg}\n")
                        return f"Error: {error_msg}"

                except json.JSONDecodeError:
                    print(data_str)
                    continue

            if result:
                return result
            else:
                return "任务完成"

        except requests.exceptions.Timeout:
            error_msg = "PhoneAgent 执行超时"
            print(f"\n❌ [PhoneAgent 错误] {error_msg}")
            return error_msg
        except Exception as e:
            error_msg = f"调用 PhoneAgent 失败: {str(e)}"
            print(f"\n❌ [PhoneAgent 错误] {error_msg}")
            return error_msg

    def chat_stream(self, user_message: str):
        """Send a message and get streaming response"""
        # Add user message
        self.messages.append({
            "role": "user",
            "content": user_message
        })

        # Call GLM API with streaming
        while True:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            payload = {
                "model": "glm-4-flash",
                "messages": self.messages,
                "tools": self.tools,
                "temperature": 0.7,
                "max_tokens": 4096,
                "stream": True  # Enable streaming
            }

            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    stream=True,
                    timeout=60
                )
                response.raise_for_status()

                # Process streaming response
                full_content = ""
                tool_calls = []
                finish_reason = None

                print("\n🤖 助手: ", end="", flush=True)

                for line in response.iter_lines():
                    if not line:
                        continue

                    line = line.decode('utf-8')
                    if not line.startswith('data: '):
                        continue

                    data_str = line[6:]  # Remove 'data: ' prefix
                    if data_str == '[DONE]':
                        break

                    try:
                        data = json.loads(data_str)
                        choice = data.get("choices", [{}])[0]
                        delta = choice.get("delta", {})
                        finish_reason = choice.get("finish_reason")

                        # Handle content
                        if "content" in delta and delta["content"]:
                            content_chunk = delta["content"]
                            full_content += content_chunk
                            print(content_chunk, end="", flush=True)

                        # Handle tool calls
                        if "tool_calls" in delta:
                            for tool_call_delta in delta["tool_calls"]:
                                index = tool_call_delta.get("index", 0)

                                # Extend tool_calls list if needed
                                while len(tool_calls) <= index:
                                    tool_calls.append({
                                        "id": None,
                                        "type": "function",
                                        "function": {
                                            "name": "",
                                            "arguments": ""
                                        }
                                    })

                                # Update tool call
                                if "id" in tool_call_delta:
                                    tool_calls[index]["id"] = tool_call_delta["id"]
                                if "type" in tool_call_delta:
                                    tool_calls[index]["type"] = tool_call_delta["type"]
                                if "function" in tool_call_delta:
                                    func_delta = tool_call_delta["function"]
                                    if "name" in func_delta:
                                        tool_calls[index]["function"]["name"] += func_delta["name"]
                                    if "arguments" in func_delta:
                                        tool_calls[index]["function"]["arguments"] += func_delta["arguments"]

                    except json.JSONDecodeError:
                        continue

                print()  # New line after streaming

                # Build assistant message
                assistant_message = {"role": "assistant"}
                if full_content:
                    assistant_message["content"] = full_content
                if tool_calls:
                    assistant_message["tool_calls"] = tool_calls

                # Add assistant message to history
                self.messages.append(assistant_message)

                # Check if tool call is needed
                if finish_reason == "tool_calls" and tool_calls:
                    # Execute tool calls
                    for tool_call in tool_calls:
                        function_name = tool_call["function"]["name"]
                        function_args = json.loads(tool_call["function"]["arguments"])

                        if function_name == "call_phoneagent":
                            result = self.call_phoneagent(function_args["prompt"])

                            # Add tool result to messages
                            self.messages.append({
                                "role": "tool",
                                "content": result,
                                "tool_call_id": tool_call["id"]
                            })

                    # Continue the conversation with tool results
                    continue

                # Return final response
                return full_content if full_content else "操作完成"

            except Exception as e:
                print(f"\n❌ GLM API 错误: {str(e)}")
                return None


def main():
    """Main interactive loop"""
    print("=" * 60)
    print("Content Agent - 命令行交互模式 (流式输出)")
    print("使用 GLM-4-Flash 模型，支持调用本地 PhoneAgent")
    print("=" * 60)

    # Get API key from environment
    api_key = os.getenv("GLM_API_KEY", "")
    if not api_key:
        print("\n❌ 错误: 请设置环境变量 GLM_API_KEY")
        print("   export GLM_API_KEY='your-api-key'")
        return

    agent = ContentAgentStream(api_key)

    print("\n💡 提示: 输入 'quit' 或 'exit' 退出")
    print("💡 示例: '给张三打电话' 或 '今天天气怎么样'\n")

    while True:
        try:
            user_input = input("👤 你: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "退出"]:
                print("\n👋 再见!")
                break

            agent.chat_stream(user_input)
            print()

        except KeyboardInterrupt:
            print("\n\n👋 再见!")
            break
        except Exception as e:
            print(f"\n❌ 错误: {str(e)}\n")


if __name__ == "__main__":
    main()
