# 🥦 Smart Vegetable Freshness Detection System

A computer vision–based vegetable classifier designed to support farmers and food supply chains in identifying **fresh** vs **rotten** vegetables. Powered by Roboflow's YOLOv11m model and deployed via FastAPI, this system enables real-time freshness detection through mobile apps, automation pipelines, or field cameras.

---

## 🚀 Key Features

✅ Uses Roboflow-trained YOLOv11m model hosted on the cloud  
✅ Detects and classifies vegetables as either **fresh** or **rotten**  
✅ Supports real-time predictions via FastAPI backend  
✅ Annotated images with bounding boxes returned as base64  
✅ Integration-ready with Flutter, ESP32, or automation scripts  
✅ Simple API call to `/detect` endpoint with image input  
✅ CORS-enabled for frontend and mobile apps  

---

## 🧠 Classes

The model is trained to detect and classify the following 8 vegetable freshness classes:

| Class             |
|------------------|
| Fresh Banana      |
| Fresh Carrot      |
| Fresh Potato      |
| Fresh Tomato      |
| Rotten Banana     |
| Rotten Carrot     |
| Rotten Potato     |
| Rotten Tomato     |

> *Additional vegetables and quality grades will be included in future model versions to expand coverage and accuracy.*

---

## ⚙️ Quickstart Guide

### 1. 📦 Install Dependencies

```bash
pip install -r requirements.txt
