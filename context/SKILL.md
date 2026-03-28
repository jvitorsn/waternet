# Skill: Research Paper → Toy Implementation

This skill transforms academic research papers into executable,
pedagogical Jupyter notebooks.

The goal is to help building notebooks based on descriptions defined by premisses from in-progress academic papers.

## Core Principle

The algorithm does not know it is a translation from concepts to code.

We preserve:
- Interfaces
- Control flow
- Decision logic
- Models
- Data

We like
- Comprehensive graphics 
- Easy readable code

## Design Rules

- Every algorithm is a pure function
- Heavy components are dependency-injected
- Intermediate states must be visible
- Qualitative trends must match the theory

## Invalid Simplifications

- Removing the paper’s core loop
- Collapsing multi-step algorithms into one step
- Hiding decisions inside black boxes