from django.urls import path
from tracker import views

app_name = 'tracker'

urlpatterns = [
    path('', views.dashboard_view, name='dashboard'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('stats/', views.stats_view, name='stats'),
    path('update_config/', views.update_config, name='update_config'),
]
