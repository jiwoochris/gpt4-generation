"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
from typing import Union


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = True):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
            
        file_name = osp.join("templates", f"{template_name}.json")
        
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
            
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )
            
    def get_description(self) -> str:
        res = self.template["description"]
        return res
    
    def get_system_message(self) -> str:
        res = self.template["system_message"]
        return res
    
    def get_user_message(self, instruction: str) -> str:
        res = self.template["user_message"].format(instruction=instruction)
        return res
    
    def get_user_message_rag(self, instruction: str, document: str) -> str:
        res = self.template["user_message"].format(instruction=instruction, document=document)
        return res