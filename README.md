# 🧭 Personal Travel Planner - Multi-Agent LLM Framework

A modular, intelligent travel planning system powered by Large Language Models (LLMs) and real-time data integration. This project uses a hierarchical **multi-agent architecture** to generate **personalized and constraint-aware itineraries** based on user preferences, availability, and external APIs.

## ✨ Features

- 🤖 **Multi-Agent Framework**: Task-specific agents handle planning, scheduling, data retrieval, and feedback.
- 🧭 **Personalized Itinerary Creation**: Incorporates user preferences like food, transport, and budget.
- 📅 **Calendar Integration**: Checks availability and schedules trips using Google Calendar.
- 🌐 **Real-Time Data**: Integrates APIs to fetch flights, accommodations, restaurants, and attractions.
- ♻️ **Iterative Plan Refinement**: Plans are refined based on user feedback and constraint satisfaction.

## 📐 System Overview

The system consists of the following key components:

- **Chatbot Agent** (Supervisor)
- **Calendar Agent** (Availability checking and scheduling)
- **User Query Builder** (Extracts structured preferences)
- **Data Retrieval Agent** (Gathers travel info via APIs)
- **Planner Module** (Generates and refines itineraries)

## 🏗️ Tech Stack

- **Python**
- **LangGraph** – for orchestrating agent workflows
- **LangChain** – for tool/API integration
- **OpenAI LLMs** – for reasoning and decision-making
