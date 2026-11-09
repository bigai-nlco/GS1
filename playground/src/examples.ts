import type { Question } from './types';
export interface Example { id: string; label: string; description: string; state: unknown; questions: Record<string, Question> }
export const examples: Example[] = [
  { id: 'choice', label: 'Multiple Choice', description: 'Choose the best team to handle a customer’s request.', state: 'My shoes arrived two weeks late and in the wrong size. Please help me return them for a refund.', questions: {
    department: { type: 'choice', instructions: 'Which team should handle this customer message?', criteria: { returns: 'Exchanges, refunds, wrong or damaged items.', billing: 'Charges, invoices, and payment problems.', shipping: 'Delivery status, delays, and lost packages.' } },
  } },
  { id: 'score', label: 'Score', description: 'Rate a product review on an ordered satisfaction scale.', state: 'The headphones sound wonderful and the battery lasts all week. The ear pads are a little tight, but I would buy them again.', questions: {
    satisfaction: { type: 'score', instructions: 'How satisfied is this customer?', criteria: ['Very dissatisfied', 'Somewhat dissatisfied', 'Neutral', 'Satisfied', 'Very satisfied'] },
  } },
  {
    id: 'resume',
    label: 'Resume screening',
    description: 'Seven Choice, Noul, and Score questions about Sasha Bernoulli’s resume.',
    state: {
      resume: `SASHA BERNOULLI
San Francisco, CA | sasha.bernoulli@email.com | github.com/sashabernoulli | linkedin.com/in/sashabernoulli

PROFESSIONAL SUMMARY
Experienced Product Engineer building developer-focused tools and platforms. Expertise in full-stack development with deep specialization in frontend architecture and UI/UX for technical audiences. Proven track record of shipping products that increase developer productivity and improve developer experience.

EXPERIENCE

Senior Product Engineer | CloudSync Systems | San Francisco, CA | Jan 2022 - Present
- Led frontend architecture redesign for cloud orchestration dashboard, reducing initial load time by 65% and improving TypeScript coverage from 42% to 98%
- Designed and implemented real-time collaboration features using WebSockets and Operational Transformation for multi-developer workflows
- Built internal API gateway and request optimization layer (Node.js/Express) that reduced backend calls by 40%, improving dashboard responsiveness
- Mentored 3 junior engineers on frontend best practices and code quality standards
- Tech Stack: React, TypeScript, Redux, Node.js, PostgreSQL, AWS

Product Engineer | DevTools Lab | San Francisco, CA | May 2021 - Dec 2021
- Architected and launched IDE plugin marketplace with 50k+ downloads; designed Vue.js frontend with Electron integration
- Implemented backend services for plugin discovery, versioning, and analytics (Python/FastAPI) handling 2M+ monthly requests
- Optimized plugin installation pipeline, reducing time from 45s to 8s through lazy-loading and caching strategies
- Built comprehensive monitoring and error tracking for frontend and backend systems using Datadog and custom logging
- Tech Stack: Vue.js, Python, FastAPI, PostgreSQL, Redis, Docker

Software Engineer | Nexus Networks | San Francisco, CA | Jan 2021 - Apr 2021
- Developed interactive network topology visualization tool using D3.js and WebGL for rendering 10k+ nodes in real-time
- Created REST API endpoints for network state management and device configuration (Go/Gin framework)
- Implemented real-time data sync between frontend and backend using gRPC, reducing latency by 50%
- Contributed to SDK documentation and developer guides for third-party integrations
- Tech Stack: React, D3.js, Go, PostgreSQL, Kubernetes

Junior Software Engineer | CodePath Systems | San Francisco, CA | Jun 2020 - Jan 2021
- Built responsive web interfaces for code analysis tools using React and CSS-in-JS
- Developed backend microservices for code parsing and analysis (Java/Spring Boot)
- Optimized database queries reducing API response times by 30%
- Tech Stack: React, JavaScript, Java, Spring Boot, MySQL

SKILLS

Frontend: React, Vue.js, TypeScript, JavaScript (ES6+), HTML5, CSS3, D3.js, WebGL, Webpack, Tailwind CSS, Material-UI
Backend: Node.js, Python (FastAPI), Go, Java (Spring Boot), SQL (PostgreSQL, MySQL), Redis, MongoDB
Developer Tools: Git, Docker, Kubernetes, GitHub Actions, Datadog, New Relic
Specializations: Developer Experience, API Design, Real-time Systems, Performance Optimization, UI/UX for Technical Users

EDUCATION

B.S. Computer Science | University of California, Berkeley | 2019
Relevant Coursework: Data Structures, Algorithms, Systems Design, Databases, Web Development

CERTIFICATIONS & ACHIEVEMENTS
- AWS Certified Solutions Architect (Associate) - 2021
- Open Source Contributor: React Query (20+ merged PRs), Electron (5+ merged PRs)
- Speaker: "Building Developer-First UI" at React Conference 2022`,
    },
    questions: {
      years_of_experience: {
        type: 'score',
        instructions: {
          question: 'How many years of professional experience does the candidate have, as of today?',
          today: 'September 15, 2026',
        },
        criteria: ['None', '2 years', '4 years', '6 years', '8 years', '10+ years'],
      },
      technical_depth: {
        type: 'score',
        instructions: "Rate hands-on engineering depth using the experience and project bullets: what the candidate personally built, how complex it was, how much they owned. Ignore skills keyword lists, titles, and company names. Score the depth shown, not the years worked. When torn between two levels, pick the lower.",
        criteria: [
          'No roles or projects where they wrote code. Technical exposure is adjacent only: manual QA, IT support, PM, sales engineering.',
          'Coding appears only as coursework, bootcamp, or tutorial projects (to-do apps, clones). Nothing shipped to real users.',
          "Small scoped work inside someone else's design: bug fixes, minor features, CRUD screens. One language, one layer. Bullets list tasks, not problems solved. Also score here if you can't tell what they actually built.",
          'Owns features end to end in a live system: designs, builds, tests, and ships with little supervision. Works across two layers (e.g. API plus frontend). Mentions code review, testing, deploys, or on-call.',
          'Owns whole systems and makes architecture tradeoffs. Depth in two domains (e.g. backend plus infrastructure). Hard problems with numbers attached: performance, scaling, migrations, incidents. Often leads projects or mentors.',
          'Deep specialist with real breadth: maintainer of a widely used open-source project, systems internals (compilers, kernels, distributed systems, database engines), or org-wide architecture ownership at significant scale.',
        ],
      },
      mentorship_demonstrated: {
        type: 'noul',
        instructions: 'Does the resume demonstrate mentoring experience?',
      },
      llm_experience: {
        type: 'noul',
        instructions: 'Does the candidate have experience developing LLM products?',
        criteria: {
          true: 'The candidate has built products or features powered by AI or Large Language Models',
          false: 'The candidate does not show experience building AI products.',
        },
      },
      certifications_opensource: {
        type: 'noul',
        instructions: 'Does the candidate have open source experience?',
      },
      career_progression: {
        type: 'choice',
        instructions: 'What type of career progression is shown?',
        criteria: {
          steady_growth: 'Clear progression with increasing seniority',
          lateral_moves: 'Similar roles at different companies',
          job_hopping: 'Frequent changes with short tenure',
          unclear: 'Progression pattern is unclear',
        },
      },
      primary_talent_profile: {
        type: 'choice',
        instructions: "Pick the best match for the candidate's talent profile. Judge from their experience hollistically, not from job titles or a skills list alone. Weight the most recent roles heaviest",
        criteria: {
          frontend_engineer: 'Builds user-facing interfaces: React, Vue, or Angular work, design systems, browser performance, accessibility. Consumes APIs but does not own them.',
          backend_engineer: 'Builds server-side services, APIs, and data models. Owns business logic, databases, queues, and service performance. Little or no UI work.',
          full_stack_engineer: 'Ships both UI and services on the same projects with neither side dominant. Not a backend engineer who occasionally edited a template.',
          mobile_engineer: 'Builds iOS, Android, or cross-platform apps (Swift, Kotlin, React Native, Flutter): app store releases, device performance, native SDKs.',
          devops_infrastructure: 'Owns how code runs and ships: CI/CD, Kubernetes, Terraform, cloud infrastructure, monitoring, reliability and on-call. Covers DevOps, SRE, and platform engineering.',
          data_engineer: 'Builds pipelines and data platforms: ETL, warehouses, Spark, Airflow, dbt, streaming. Serves analysts and models rather than end users.',
          ml_ai_engineer: 'Trains, fine-tunes, evaluates, or serves models. Includes applied ML, LLM, and research engineering.',
          security_engineer: 'Application, cloud, or product security: threat modeling, penetration testing, detection engineering, identity, vulnerability remediation.',
          embedded_systems: 'Low-level work: firmware, drivers, kernels, compilers, robotics, or hardware-constrained C, C++, and Rust.',
          other: 'Real engineering that fits none of the above, such as QA automation, game development, or forward-deployed and solutions engineering.',
        },
      },
    },
  },
];
