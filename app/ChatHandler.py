import json
from utils.route import route
from utils.log import logger

import tornado.web
import requests
import traceback
from . import ModelHandler

ModelHandler.init_model()

retry_times = 3
global_dict = {}


@route("/")
class ChatHandler(tornado.web.RequestHandler):
    webhook = ""

    def get(self):
        return self.write_json({"ret": 200})

    def post(self):
        try:
            request_data = self.request.body
            data = json.loads(request_data)
            print(data)
            prompt = data['text']['content']
            self.webhook = data['sessionWebhook']
            if "/clear" in prompt:
                self.clear_context(data)
                self.notify_dingding('已清空上下文')
                return self.write_json({"ret": 200})

            for i in range(retry_times):
                try:
                    resp, history = ModelHandler.predict(prompt)
                    print(resp)
                    break
                except:
                    traceback.print_exc()
                    continue
                logger.info(f"pass")
            self.notify_dingding(resp)
        except:
            traceback.print_exc()
            return self.write_json({"ret": 500})
        history += history

    def get_context(self, data):
        storeKey = self.get_context_key(data)
        if (global_dict.get(storeKey) is None):
            global_dict[storeKey] = []
        return global_dict[storeKey]

    def get_context_key(self, data):
        conversation_id = data['conversationId']
        sender_id = data['senderId']
        return conversation_id + '@' + sender_id

    def set_context(self, data, response):
        prompt = data['text']['content']
        storeKey = self.get_context_key(data)
        if (global_dict.get(storeKey) is None):
            global_dict[storeKey] = []
        global_dict[storeKey].append({"role": "user", "content": prompt})
        global_dict[storeKey].append(
            {"role": "assistant", "content": response})

    def clear_context(self, data):
        store_key = self.get_context_key(data)
        global_dict[store_key] = []

    def write_json(self, struct):
        self.set_header("Content-Type", "application/json; charset=UTF-8")
        self.write(tornado.escape.json_encode(struct))

    def notify_dingding(self, answer):
        data = {
            "msgtype": "text",
            "text": {
                "content": answer
            },

            "at": {
                "atMobiles": [
                    ""
                ],
                "isAtAll": False
            }
        }

        try:
            r = requests.post(self.webhook, json=data)
            reply = r.json()
            logger.info("dingding: " + str(reply))
        except Exception as e:
            logger.error(e)