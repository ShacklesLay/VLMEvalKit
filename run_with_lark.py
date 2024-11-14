from run import main
from vlmeval.smp import *

import sys
sys.path.append("/remote-home1/cktan/server_tools/")
from larknotice import lark_sender

@lark_sender(webhook_url="https://open.feishu.cn/open-apis/bot/v2/hook/9824a4f2-07e2-40cc-ae32-74ded5a0db96")
def run_with_lark():
    load_env()
    main()
    
if __name__ == "__main__":
    run_with_lark()