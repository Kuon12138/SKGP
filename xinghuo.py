curl -i -k -X POST 'https://spark-api-open.xf-yun.com/v1/chat/completions' \
--header 'Authorization: Bearer 123456' \#注意此处把“123456”替换为自己的APIPassword
--header 'Content-Type: application/json' \
--data '{
    "model":"generalv3.5",lite,generalv3,pro-128k,generalv3.5,max-32k
4.0Ultra
    "messages": [
        {
            "role": "user",
            "content": "来一个只有程序员能听懂的笑话"
        }
    ],
    "stream": true
}'
{
    "model": "generalv3.5",
    "user": "用户唯一id",
    "messages": [
        {
            "role": "system",
            "content": "你是知识渊博的助理"
        },
        {
            "role": "user",
            "content": "你好，讯飞星火"
        }
    ],
    // 下面是可选参数
    "temperature": 0.5,取值范围[0, 2] 默认值1.0
    "top_p": 1,取值范围(0, 1] 默认值1
    "top_k": 4,取值范围[1, 6] 默认值4
    "stream": false,
    "max_tokens": 1024,Pro、Max、Max-32K、4.0 Ultra 取值为[1,8192]，默认为4096;Lite、Pro-128K 取值为[1,4096]，默认为4096。
    "presence_penalty": 1,
    "frequency_penalty": 1,
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "str2int",
                "description": "将字符串类型转为 int 类型",
                "parameters": {...} // 需要符合 json schema 格式
            }
        },
        {
            "type": "web_search",
            "web_search": {
                "enable": true
                "show_ref_label":true
                "search_mode":"deep" // deep:深度搜索 / normal:标准搜索,不同的搜索策略，效果不同，并且token消耗也有差异
            }
        }
    ],
    "response_format": {
        "type": "json_object"
    },
    "suppress_plugin": [
        "knowledge"
    ]
}