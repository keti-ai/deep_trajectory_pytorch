__copyright__ = """

    Copyright 2023 YongHyeok Seo

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    github : https://github.com/syh4661
    
"""
__license__ = "Apache 2.0"


from cfg._configs import Parser
from trainer import Trainer
# from model.base_model import Model_example
import model



if __name__ == '__main__':
    arg_=Parser()
    trainer=Trainer(model=model.seresnext50_32x4d())
    trainer.train()
