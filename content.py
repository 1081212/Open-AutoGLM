#!/usr/bin/env python3
"""
Simple Content Agent with GLM-4.7-flash
Supports calling local phoneagent service at localhost:8000/run
"""

import os
import json
import requests
from typing import List, Dict, Any


class ContentAgent:
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
        """Call the local phoneagent service"""
        print(f"\n🤖 [调用 PhoneAgent] 参数: {prompt}")
        try:
            response = requests.post(
                self.phoneagent_url,
                json={"prompt": prompt},
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            print(f"✅ [PhoneAgent 返回] {json.dumps(result, ensure_ascii=False)}")
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            error_msg = f"调用 PhoneAgent 失败: {str(e)}"
            print(f"❌ [PhoneAgent 错误] {error_msg}")
            return error_msg

    def chat(self, user_message: str) -> str:
        """Send a message and get response"""
        # Add user message
        self.messages.append({
            "role": "user",
            "content": user_message
        })

        # Call GLM API
        while True:
            response = self._call_glm_api()

            if not response:
                return "API 调用失败"

            choice = response["choices"][0]
            message = choice["message"]
            finish_reason = choice["finish_reason"]

            # Add assistant message to history
            self.messages.append(message)

            # Check if tool call is needed
            if finish_reason == "tool_calls" and "tool_calls" in message:
                # Execute tool calls
                for tool_call in message["tool_calls"]:
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
            return message.get("content", "")

    def _call_glm_api(self) -> Dict[str, Any]:
        """Call GLM API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": "glm-4.7-flash",
            "messages": self.messages,
            "tools": self.tools,
            "temperature": 0.7,
            "max_tokens": 4096
        }

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ GLM API 错误: {str(e)}")
            return None


def main():
    """Main interactive loop"""
    print("=" * 60)
    print("Content Agent - 命令行交互模式")
    print("使用 GLM-4.7-flash 模型，支持调用本地 PhoneAgent")
    print("=" * 60)

    # Get API key from environment
    api_key = os.getenv("GLM_API_KEY", "")
    if not api_key:
        print("\n❌ 错误: 请设置环境变量 GLM_API_KEY")
        print("   export GLM_API_KEY='your-api-key'")
        return

    agent = ContentAgent(api_key)

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

            print("\n🤔 思考中...")
            response = agent.chat(user_input)
            print(f"\n🤖 助手: {response}\n")

        except KeyboardInterrupt:
            print("\n\n👋 再见!")
            break
        except Exception as e:
            print(f"\n❌ 错误: {str(e)}\n")


if __name__ == "__main__":
    main()
