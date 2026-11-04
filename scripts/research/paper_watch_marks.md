# Task: mark each paper in the report file

The prompt gives the path of the report file. Use only the title and the abstract in that file. Do not fetch web pages. This task replaces the report format of the shared rules with the format below.

## Marks

- **to compare:** the thesis must compare with this paper. It is text-to-motion with autoregressive, streaming or token generation, or it reports HumanML3D numbers.
- **relevant:** the same field. The thesis can cite it or the owner can read it.
- **not relevant:** all other papers.

## Output

1. A table with one row for each paper: `| id | mark | reason |`. The reason has at most 15 words.
2. A section "Novelty threats". List each paper that may be a discrete, causal, token-autoregressive Mamba or state-space text-to-motion generator. Give one line of reason for each. If there is none, write "none".

Maximum 1500 words.
