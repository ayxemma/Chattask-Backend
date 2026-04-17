import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
