"""URL routes for the interpreter app (browser-camera architecture)."""

from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("process", views.process, name="process"),
    path("depth", views.depth, name="depth"),
    path("status", views.status, name="status"),
    path("record/start", views.record_start, name="record_start"),
    path("record/stop", views.record_stop, name="record_stop"),
    path("record/cancel", views.record_cancel, name="record_cancel"),
    path("transcript/clear", views.clear, name="clear"),
    path("lexicon", views.lexicon, name="lexicon"),
    path("vocab/add", views.vocab_add, name="vocab_add"),
    path("vocab/delete", views.vocab_delete, name="vocab_delete"),
]
