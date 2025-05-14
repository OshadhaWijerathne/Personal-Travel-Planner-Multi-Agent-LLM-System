# 🧭 Personal Travel Planner – Multi-Agent LLM System

A modular, intelligent travel planning system powered by Large Language Models (LLMs) and structured data integration. This project uses a hierarchical multi-agent architecture to generate personalized and constraint-aware itineraries based on user preferences, availability, and pre-stored datasets.

---

## ✨ Features

- 🤖 **Multi-Agent Framework**: Task-specific agents handle planning, scheduling, data retrieval, and feedback.
- 🧭 **Personalized Itinerary Creation**: Incorporates user preferences like food, transport, and budget.
- 📅 **Calendar Integration**: Checks availability and schedules trips using Google Calendar.
- 📊 **Structured Data Integration**: Utilizes curated, pre-stored datasets for flights, accommodations, restaurants, and attractions.
- ♻️ **Iterative Plan Refinement**: Plans are refined based on user feedback and constraint satisfaction.
- 🚀 **API Interface**: FastAPI provides a lightweight and scalable backend to serve and interact with the planning system.

---

## 📐 System Overview

The system consists of the following key components:

- **Chatbot Agent** – Supervises the entire planning process.
- **Calendar Agent** – Checks calendar availability and schedules trips.
- **User Query Builder** – Extracts structured preferences from user input.
- **Data Retrieval Agent** – Gathers travel info from stored datasets.
- **Planner Module** – Generates and iterates on itineraries.
- **FastAPI Backend** – Exposes endpoints for frontend or external use.

---

## 🏗️ Tech Stack

- **Python**
- **FastAPI** – Backend API framework
- **LangGraph** – Orchestrates agent workflows
- **LangChain** – For dataset/tool integration
- **OpenAI LLMs** – Powers reasoning and decision-making

---
