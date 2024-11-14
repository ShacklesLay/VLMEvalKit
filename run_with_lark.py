from run import main
from vlmeval.smp import *
from torch_npu.contrib import transfer_to_npu

import sys
sys.path.append("/home/image_data/cktan/reps/server_tools")
from larknotice import lark_sender

@lark_sender(webhook_url="https://open.feishu.cn/open-apis/bot/v2/hook/893b2fc5-1a17-4c8b-90e7-e2d5e4d1a846")
def run_with_lark():
    load_env()
    main()
    
if __name__ == "__main__":
    run_with_lark()