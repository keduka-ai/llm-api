from django.urls import path
from .views import ChatCompletionsView, TextPromptView

urlpatterns = [
    path("chat/completions/", ChatCompletionsView.as_view(), name="chat-completions"),
    path("text-prompt/", TextPromptView.as_view(), name="text-prompt"),
]
