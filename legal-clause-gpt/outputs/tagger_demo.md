# LegalClauseTagger demo (`gpt2_n60k`, review threshold 0.881)

## Hand-written clauses

| clause | tag | confidence | review? | runner-up |
|---|---|---|---|---|
| This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware, without regard to its conflict of la... | Governing Laws | 0.975 | no | Applicable Laws (0.023) |
| If any provision of this Agreement is held invalid or unenforceable, the remaining provisions shall continue in full force and effect. | Severability | 0.996 | no | Enforceability (0.002) |
| The Employee shall be entitled to twenty days of paid vacation per calendar year. | Vacations | 1.000 | no | Expenses (0.000) |
| The parties shall cooperate in good faith. | Cooperation | 0.988 | no | Further Assurances (0.009) |

## Real test clauses

| case | clause | true type | tag | confidence | review? |
|---|---|---|---|---|---|
| most confident, correct | The Company shall not sell, offer for sale or solicit offers to buy or otherwise negotiate in respect of any security (as defined in Section... | Integration | Integration | 1.000 | no |
| flagged for review, tag was wrong (caught) | No failure or delay on the part of AMSC or Subcontractor in the exercise of any power, right or privilege, and no course of dealing between ... | No Waivers | Waivers | 0.600 | needs review |
| flagged for review, tag was right anyway | The Board of Directors may appoint from among its members committees, composed of one or more directors, to serve at the pleasure of the Boa... | Qualifications | Qualifications | 0.600 | needs review |
| confidently wrong | In the event of Participant’s death prior to the Vesting Date, the Option will fully vest and the underlying Shares shall be issued to his/h... | Vesting | Death | 1.000 | no |

Consistency check: tagger confidence on CPU vs saved Colab logits differ by at most 0.0004 (fp16 on the GPU vs fp32 here).
