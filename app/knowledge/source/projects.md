# Projects

## LetsGo South Africa - AI-Powered Tourism Platform

- Website: https://letsgodb.web.app/
- Period: Apr 2026 to Present
- Associated with: Zeno
- Problem solved: Provides a modern digital platform for managing tourism
  packages, improving package discovery, and supporting customer enquiries
  through both forms and AI-assisted chat.
- Key features:
  - Public travel package browsing
  - Admin package creation, editing, publishing, and image management
  - Online enquiry workflow
  - AI travel assistant for package and travel-related questions
  - Authentication-protected admin workflows
  - Cloud-hosted frontend and backend deployment
- Technical highlights:
  - FastAPI backend
  - PostgreSQL database
  - React/Vite frontend
  - Authentication and admin access control
  - Image handling
  - Enquiry workflows
  - Cloud deployment
- Technologies: FastAPI, PostgreSQL, React, Vite, Firebase Hosting, GCP Cloud
  Run, Cloud SQL, Docker, GitHub Actions

## BeautyVerse - Beauty Services Marketplace

- Website: https://beautyverse.co.za/
- Type: Marketplace and full-stack web application
- Problem solved: Enables providers to create and manage service listings while
  customers browse beauty services and submit enquiries.
- Key features:
  - Provider-owned listing creation and management
  - Public browsing for beauty services
  - Customer enquiry submission
  - Reusable marketplace foundation with listings, categories, users, images,
    and enquiries
- Technical highlights:
  - FastAPI backend services
  - PostgreSQL with SQLAlchemy and Alembic
  - Docker-based local development
  - Authentication and role-based access patterns
  - Frontend flows for marketplace discovery and provider management
- Technologies: FastAPI, PostgreSQL, SQLAlchemy, Alembic, Docker

## MedDesk - AI Clinical Intake Proof of Concept

- Website: https://meddesk.co.za/
- Problem solved: Helps structure patient intake before consultation, reducing
  unstructured handover information and giving clinicians a reviewable draft
  rather than an automated diagnosis.
- Key features:
  - Patient-facing AI symptom intake chat
  - Conversation persistence
  - Subjective and patient-reported Objective information collection
  - Red-flag symptom detection
  - Structured draft SOAP note generation
  - Clinician review workflow
- Technical highlights:
  - AI-assisted intake orchestration
  - Safety-aware red-flag escalation logic
  - Structured clinical note drafting
  - Backend APIs for intake sessions and messages
  - Designed as clinician-support software rather than diagnostic automation
- Technologies: FastAPI, Python, LLM APIs, PostgreSQL, prompt orchestration,
  Docker
- Important boundary: This system is a proof of concept for intake support and
  clinician review. It is not a replacement for professional medical judgment
  and should not be represented as a diagnostic system.

## Personal Website Digital Twin / Portfolio AI Assistant

- Problem solved: Allows visitors to interactively explore Tumelo's experience
  and projects instead of only reading a static CV.
- Key features:
  - Grounded profile-based responses
  - Conversation persistence
  - Professional tone and response guidelines
  - Backend service orchestration
  - Portfolio integration
- Technical highlights:
  - FastAPI backend
  - Prompt builder and service orchestration
  - Structured profile and summary files used as grounding inputs
  - Conversation storage
- Technologies: FastAPI, Python, OpenAI LLM APIs, JSON knowledge base, Docker
