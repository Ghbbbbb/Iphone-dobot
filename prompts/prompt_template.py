import os

try:
    from langchain.prompts.chat import PromptTemplate
except:
    from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

class PromptLoader:
    def __init__(self, prompt_doc: str) -> None:
        # 获取当前文件所在目录
        current_dir = os.path.dirname(__file__)
        # 构建base目录路径
        self.dir_path = os.path.join(current_dir, prompt_doc)

    @property
    def pri_prompt(self) -> str:
        with open(os.path.join(self.dir_path, "primitives.txt"),encoding='utf-8') as f:
            prompt = f.read()
        return prompt

    @property
    def sce_prompt(self) -> str:
        with open(os.path.join(self.dir_path, "scene.txt"),encoding='utf-8') as f:
            prompt = f.read()
        return prompt

    @property
    def sys_prompt(self) -> str:
        with open(os.path.join(self.dir_path, "system.txt"),encoding='utf-8') as f:
            prompt = f.read()
        return prompt
    
    @property
    def settings_prompt(self) -> str:
        with open(os.path.join(self.dir_path, "task_settings.txt"),encoding='utf-8') as f:
            prompt = f.read()
        return prompt
    


def get_qa_template_baichuan_dobot(prompt_doc: str):
    prompt_loader = PromptLoader(prompt_doc)

    _ROBOT_PROMPT_TEMPLATE = f"""
    {prompt_loader.sys_prompt}

    {prompt_loader.sce_prompt}

    {prompt_loader.pri_prompt}

    """


    ##################有RAG#####################
    _DEFAULT_QA_TEMPLATE_BAICHUAN_DOBOT = _ROBOT_PROMPT_TEMPLATE + "{context}" + """
Use the above context to answer the user's question and to perform the user's command.
-----------
Human: {question}
You:"""
    
    QA_TEMPLATE_BAICHUAN = PromptTemplate(
        input_variables=["context", "question"],
        template=_DEFAULT_QA_TEMPLATE_BAICHUAN_DOBOT,
    )

    return QA_TEMPLATE_BAICHUAN

