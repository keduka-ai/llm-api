from django.db import models
from django.utils import timezone

# from django.core.validators import MaxLengthValidator
import hashlib
import numpy as np
from django.core.exceptions import ValidationError
# from utilities.embeddings import get_embeddings, numpy_to_bytes, bytes_to_numpy
import config
from django.core.validators import MaxLengthValidator


class AgentMemory(models.Model):
    prompt_id = models.CharField(
        max_length=64,
        editable=False,
        verbose_name="Hashed Topic ID",
        db_index=True,
    )
    prompt = models.TextField(verbose_name="Prompt", blank=False)
    subject = models.CharField(max_length=512, db_index=True, verbose_name="Subject")
    response = models.TextField(verbose_name="Response", blank=False)
    prompt_vector = models.BinaryField(verbose_name="Prompt Vector")
    response_vector = models.BinaryField(verbose_name="Response Vector")
    created_date = models.DateTimeField(
        default=timezone.now, verbose_name="Created Date"
    )
    modified_date = models.DateTimeField(
        auto_now=True, verbose_name="Last Modified Date"
    )

    def hash_prompt(self, prompt):
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def bytes2numpy(self, byte_arr):
        if len(byte_arr) == 0:
            raise ValueError("Byte array is empty")
        return np.frombuffer(byte_arr, dtype=np.float32).reshape(-1, config.D_MODEL)

    def validate_fields(self):
        if not self.subject:
            raise ValidationError({"subject": "The subject cannot be empty."})
        if not self.prompt:
            raise ValidationError({"prompt": "The prompt cannot be empty."})
        if not self.prompt_vector or not self.response_vector:
            raise ValidationError(
                {"vectors": "Prompt vector and response vector must not be empty."}
            )

    def __str__(self):
        return f"AgentMemory(prompt_id={self.prompt_id}, subject='{self.subject}')"

    class Meta:
        verbose_name = "AgentMemory"
        verbose_name_plural = "Knowledge Graphs"
        db_table = "octagent_memory"
