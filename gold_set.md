# Gold set — vérité terrain du corpus

14 questions, chacune rattachée à sa police, à la clause qui répond, et au comportement attendu.
Sert à mesurer le retrieval (la bonne clause est-elle dans le top-k ?) séparément de la réponse finale.

| # | Question | Type | Police(s) | Clause | Comportement attendu |
|---|----------|------|-----------|--------|----------------------|
| 1 | Quelle est la date d'effet du contrat AUTO-2024-0137 ? | Lookup | #1 | en-tête | « 1er mars 2024 » — top-1 suffit |
| 2 | Quelle est la franchise bris de glace de mon assurance auto ? | Désambiguïsation | #1 / #2 | §5 | Deux autos, valeurs différentes (0 CHF vs 150 CHF) → demander de quel contrat il s'agit, ne pas trancher |
| 3 | Toutes les exclusions applicables à la garantie Vol du contrat AUTO-2024-0137 ? | Liste (M2A) | #1 | §3 + §4 + avenant §8 | Réponse complète = les trois blocs, dont l'avenant. Un parcours séquentiel s'arrête trop tôt |
| 4 | Si je rentre seul dans un mur avec ma voiture, suis-je couvert par AUTO-2024-0137 ? | Distracteur (M3) | #1 | §2.2 | Dommages tous accidents, y compris sans tiers. Ne PAS renvoyer la RC (§2.1) |
| 5 | Un dégât des eaux dû à une canalisation gelée est-il couvert ? | Contradiction (C1) | #3 vs #4 | §2.2 | Dépend du contrat : #3 oui, #4 non. Réponse unique = faux |
| 6 | Parmi les contrats auto et moto, lequel a la franchise collision la plus basse ? | Comparaison | #1 / #2 / #7 | §5 | #1 (500 CHF). Piège : #2 n'a pas de garantie collision, #7 = 750 CHF |
| 7 | De combien de temps je dispose pour déclarer un sinistre ? | Contradiction (C4) | #1 / #3 / #5 | §7 (et §7 hab) | Dépend du contrat et du péril : auto 5 j ouvrables, dégât des eaux 48 h, santé 30 j |
| 8 | Comment fonctionne la franchise de ma complémentaire santé ? | Vocabulaire (D) | #5 | §1, §3 | Franchise annuelle (par année civile), ≠ franchise par sinistre des autos. Ne pas confondre avec le délai de carence |
| 9 | Quelle police habitation était en vigueur le 15 avril 2024 ? | Temporel (E) | #3 | en-tête | #3 uniquement (#4 débute le 01/07/2024) |
| 10 | Quelle police habitation offre le plafond dégâts des eaux le plus élevé ? | Comparaison inter-docs | #3 vs #4 | §5 | #4 (75 000 CHF) contre #3 (50 000 CHF) |
| 11 | Quel est le plafond de remboursement optique du contrat AUTO-2024-0137 ? | Absence (M4) | — | — | S'abstenir : information non présente dans la police auto |
| 12 | En voyage, suis-je couvert si mon vol est annulé à cause d'une pandémie ? | Absence (M4) | — | — | S'abstenir : le contrat #6 ne traite pas ce cas |
| 13 | Quelle est la franchise dégâts des eaux du contrat HAB-2024-0461 ? | Avenant (M2B) | #4 | §8 (avenant) | 250 CHF après avenant, pas 500 CHF (§5). Le retrieval doit atteindre l'avenant |
| 14 | Quelle est la franchise vol de ma moto ? | Quasi-jumeau | #7 | §5 | #7 = 500 CHF. Ne pas renvoyer une auto |

## Comment t'en servir pour la mesure
- **Recall@k du retrieval** : pour chaque question, la clause listée ci-dessus est-elle dans les k passages récupérés ? (Questions 11 et 12 : le bon comportement est qu'AUCUN passage pertinent ne remonte avec un score élevé.)
- **Exactitude de la réponse** : la réponse finale correspond-elle au comportement attendu ? Compare le taux de bonnes réponses au Recall@k — l'écart entre les deux te dit si la chaîne casse au retrieval ou à la génération.
