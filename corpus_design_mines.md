# Carte du corpus — où chaque difficulté est enterrée

Document de travail (à NE PAS déployer avec le chatbot). Il te sert à savoir exactement
ce que tu as construit, pour défendre chaque choix devant Kezhan et rédiger ta note finale.
Principe : aucune difficulté n'est un piège artificiel — chacune reproduit un trait réel d'un contrat.

## Les 4 modes d'échec visés
- **M1 — lookup propre** : réponse dans une clause unique et claire. Cas de contrôle (le système DOIT réussir).
- **M2 — liste éparpillée** : la réponse complète est répartie sur plusieurs endroits, dont un avenant en fin de document. Casse `complete_answer_found` en mode séquentiel.
- **M3 — distracteur sémantique** : la bonne clause est formulée autrement que la question ; des clauses proches lexicalement mais hors-sujet remontent d'abord. C'est le « ressembler ≠ contenir » de l'article 1.
- **M4 — absence** : la réponse n'existe dans aucune police. Bon comportement = s'abstenir, pas inventer.

## Ce qui est planté dans chaque police
- **#1 AUTO tous risques** — police de référence. Porte **M2A** (avenant §8 qui AJOUTE une couverture + une exclusion). Porte **M3** via §2.2 « y compris en l'absence de tiers ».
- **#2 AUTO tiers étendu** — quasi-jumeau lexical de #1, mais PAS de garantie collision, franchises différentes (vol 750, bris 150), zone plus étroite, exclusion propre (conducteur non désigné). Sert la désambiguïsation et le piège « même terme, valeurs différentes ».
- **#3 HAB multirisque** — **couvre** la rupture de canalisation par gel (§2.2). Même date d'effet que #1 (01/03/2024). Délai de déclaration dégât des eaux = 48 h.
- **#4 HAB éco** — faux jumeau de #3 : **exclut** le gel (§2.2), plafonds/zone différents, clause spécifique énergie renouvelable. Porte **M2B** (avenant §8 qui MODIFIE une valeur existante : franchise 500 → 250).
- **#5 SANTÉ** — change de domaine mais garde le vocabulaire : « franchise » = annuelle, + « délai de carence » (terme voisin qui n'est pas une franchise). Délai de déclaration = 30 j.
- **#6 VOYAGE** — exclusion géographique formulée comme celle de l'auto (distracteur inter-docs). Ne mentionne AUCUNE pandémie → sert **M4**.
- **#7 MOTO** — quasi-jumeau des autos, phrases presque identiques. Franchises propres (collision 750, vol 500). Date 01/03/2023 (proche mais distincte des 01/03/2024).

## Relations entre documents (les pièges inter-contrats)
- **R1 — dates proches** : #1 et #3 = 01/03/2024 ; #7 = 01/03/2023. Une question « auto » ne doit pas ramener l'habitation.
- **R2 — famille véhicule** : #1, #2, #7 partagent RC/vol/incendie. « Ma franchise auto » a deux candidats légitimes (#1, #2) + un leurre (moto).
- **R3 — vocabulaire partagé** : « franchise » signifie par-sinistre (auto) vs annuelle (santé). Piège de sens.
- **R4 — zones géographiques** : exclusion géo auto (§3d) ~ exclusion géo voyage (#6 §3c). Distracteur sémantique entre docs.
- **C1 — gel** : #3 couvre / #4 exclut. Même péril, réponse inverse.
- **C2 — franchises** : collision #1=500, #7=750, #2=non couvert.
- **C3 — conditions du vol** : #1 exige un antivol ; #2 exige un stationnement fermé la nuit.
- **C4 — délai de déclaration** : 5 j ouvrables (auto) / 48 h (dégât des eaux) / 30 j (santé). « Combien de temps ai-je ? » n'a pas de réponse unique.

## Subtilité à connaître (et à exploiter ou non, à ton choix)
Les deux habitations **se chevauchent** : #3 court jusqu'au 28/02/2025 et #4 démarre le 01/07/2024. Donc à une date d'été 2024, les DEUX sont en vigueur. La question temporelle du gold set (15 avril 2024) évite volontairement ce chevauchement pour avoir une réponse unique. Tu peux, si tu veux un test plus dur, ajouter « quelle(s) police(s) le 15 septembre 2024 ? » dont la vraie réponse est « les deux » — bon test de raisonnement, mais assure-toi que ton système est censé le gérer avant de le noter comme un échec.

## Le fil rouge (rappel)
Tu as généré ces documents, donc tu possèdes la vérité terrain — c'est ce qui rend le gold set possible et honnête.
Relis chaque police au moins une fois : tu dois pouvoir dire, sans ce fichier sous les yeux, où chaque mine est enterrée.
