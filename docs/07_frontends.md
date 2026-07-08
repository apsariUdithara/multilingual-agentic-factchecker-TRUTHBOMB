# 07: Frontends

The project provides two distinct user interfaces for interacting with the fact-checking engine.

## 1. Streamlit Dashboard (`app.py`)

A lightweight, Python-based interface for rapid testing and internal use.

### Features
- **Large Text Area**: For pasting social media posts.
- **Verification Trigger**: Sends a POST request to the FastAPI `/api/v1/verify` endpoint.
- **Result Visualization**:
  - Displays the verdict with color-coded status (Green for TRUE, Red for FALSE, Grey for UNVERIFIED).
  - Shows the normalized English claim.
  - Displays the reasoning provided by the LLM.
- **Context Inspector**: An expandable section showing the exact snippets retrieved from the database, along with their source and distance scores.

## 2. React Web App (`frontend/`)

A modern, production-ready frontend built with **React**, **Vite**, and **Tailwind CSS**.

### Components (`frontend/src/components/`)
- **`FactCheckerForm.tsx`**: The main input component where users submit their claims.
- **`ResultCard.tsx`**: A polished UI component to display the verdict and confidence metrics.
- **`ContextSources.tsx`**: A detailed view of the evidence used for the verification.
- **`ui/` (Shadcn/ui)**: Reusable UI primitives like buttons, textareas, and badges.

### State Management
Uses standard React hooks (`useState`, `useEffect`) to manage API interactions and UI state. It connects to the same FastAPI backend as the Streamlit app.
