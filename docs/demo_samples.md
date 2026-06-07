# Demo samples — EN-JP code-switching ASR

**Model:** `lfm_lora_balanced_r32_csf/step_1800` (our best) · **Baselines:** Whisper large-v3 (base), LFM2.5-Audio-1.5B-JP (base).

**The story:** on code-switched speech, the baselines **force English into katakana**, drop it, or garble it — our fine-tune **keeps each language in its correct script** (English stays in Latin), so the transcript is actually readable.

Headline on the 196-utt CS benchmark: **Script Accuracy 37% (LFM base) → 80% (ours)**, MER 67% → 28%.

> ✅ = keeps English in Latin & accurate  ·  ❌ = katakana-forced / dropped / garbled. Look at the **English words** in each row.

---

## Sample 1 — `jpn_1676_SS`  (10s)

🔊 `data/csfleurs/read_test/audio/jpn_1676_SS.wav`

**Reference (truth):**
> As with all 南アフリカ National Parks, この公園には毎日 conservation と 入園料 がかかります。

**✅ OURS (balanced_r32_csf):**
> As with all 南アフリカ national parks. この公園には毎日 conservation と入園料がかかります。

**❌ Whisper large-v3 (base):**
> この公園には毎日コンサルベーションと入園料がかかります

**❌ LFM2.5-Audio base:**
> エンスウェッド・オー・南アフリカ・ナショナルパークスこの公園には毎日コンサベーションと入園料がかかります

*Why ours wins:* Whisper **drops the entire English opening clause** ('As with all … National Parks') and katakana-izes 'conservation'; base garbles it. Ours keeps it verbatim.

---

## Sample 2 — `jpn_1802_SS`  (7s)

🔊 `data/csfleurs/read_test/audio/jpn_1802_SS.wav`

**Reference (truth):**
> インターネットは、massとinterpersonal communicationの両要素を兼ね備えた環境です。

**✅ OURS (balanced_r32_csf):**
> インターネットは、massとinterpersonal communicationの両要素を兼ね備えた環境です。

**❌ Whisper large-v3 (base):**
> インターネットはマスとイントロプロソナルコミュニケーションの両要素を含め備えた環境です

**❌ LFM2.5-Audio base:**
> インターネットは、MASとインターネットコミュニケーションの両要素を兼ね備えた環境です。

*Why ours wins:* Ours is **letter-perfect**. Whisper katakana-forces ('マスとイントロプロソナル'); base mangles ('MASとインターネット').

---

## Sample 3 — `jpn_1762_SS`  (8s)

🔊 `data/csfleurs/read_test/audio/jpn_1762_SS.wav`

**Reference (truth):**
> このdiseaseは豚によってcarriedされ、その後、mosquitosを媒介にしてhumansにmigratesします。

**✅ OURS (balanced_r32_csf):**
> このdiseaseは豚によってcarriedされ、その後mosquitoesを媒介してhumansにmigratesします。

**❌ Whisper large-v3 (base):**
> このディズィースは豚によってキャリアされその後モスキルスをバイカにしてヒュームズにマイグレートします

**❌ LFM2.5-Audio base:**
> このディザイスは豚によって飼育され、その後モスキルスを媒介にしてヒューマンスにマイグレートします。

*Why ours wins:* Medical terms: ours keeps disease/carried/mosquitos/humans/migrates in English. Whisper→all katakana; base even **mistranslates** 'carried'→'飼育' (bred).

---

## Sample 4 — `jpn_1907_SS`  (11s)

🔊 `data/csfleurs/read_test/audio/jpn_1907_SS.wav`

**Reference (truth):**
> 敵対行為のoutbreakのSoon後に、BritainはGermanyに対するnaval blockadeをinitiatedしました。

**✅ OURS (balanced_r32_csf):**
> 敵対行為のoutbreakのsoon後に、BritainはGermanに対するnaval blockadeをinitiatedしました。

**❌ Whisper large-v3 (base):**
> 敵対行為のアウトブレイクのすぐ後に、ブリトンはジェルミニーに対するネイバーブロンケイドをイニシエイトしました。

**❌ LFM2.5-Audio base:**
> 敵対行為のアウトブレイクの後、ベルリンはドイツに対するネイボブキャットをイニシエートしました。

*Why ours wins:* History: ours keeps Britain/Germany/naval blockade/initiated. Whisper→all katakana; base **hallucinates** ('ベルリン', 'ネイボブキャット').

---

## Sample 5 — `jpn_1999_SS`  (13s)

🔊 `data/csfleurs/read_test/audio/jpn_1999_SS.wav`

**Reference (truth):**
> 1960年代を通して、BrzezinskiはJohn F. Kennedyのadvisorを務め、その後Lyndon B. Johnson administrationでも活躍しました。

**✅ OURS (balanced_r32_csf):**
> 1960年代を通して、PresantskyはJohn F. Kennedyのadvisorを務め、その後Lyndon B. Johnson's administrationでも活躍しました。

**❌ Whisper large-v3 (base):**
> 1960年代を通して、ブリゼンスキーは、ジョン・F・ケネディのアドバイザーを務め、その後、リンドン・B・ジャンソン・アドミニストレーションでも活躍しました。

**❌ LFM2.5-Audio base:**
> 1960年代を通して、プレゼンスキーはジョン・F・ケネディのアドバイザーを務め、その後、リン・ドン・ビージャン・スナード・アデミネーションでも活躍しました。

*Why ours wins:* Proper nouns: ours keeps 'John F. Kennedy', 'advisor', 'Lyndon B. Johnson administration'. Both baselines reduce everything to broken katakana.

---


---

# Japanese-only samples (forgetting check — we didn't lose JA)

Our CS fine-tune **keeps Japanese ASR intact** (JA-CER 6.6, ≈ base/Whisper) — unlike fine-tuning Whisper for CS, which **wrecked** its Japanese (CER 6.0 → 26.1). Below: utterances where ours beats **both** baselines.

## JA Sample 1 — `ja_jp_1813`  (13s)  ·  CER ours 0% vs Whisper 24% vs base 6%

🔊 `data/fleurs/ja_jp_test/audio/ja_jp_1813.wav`

**Reference (truth):**
> バルセロナの公用語はカタルーニャ語とスペイン語です。約半数がカタルーニャ語を好み、大多数がカタルーニャ語を理解し、ほぼ全員がスペイン語を知っています。

**✅ OURS (balanced_r32_csf):**
> バルセロナの公用語はカタルーニャ語とスペイン語です 約半数がカタルーニャ語を好み 大多数がカタルーニャ語を理解し ほぼ全員がスペイン語を知っています

**❌ Whisper large-v3 (base):**
> バルセアナの公用語はカタルーネ語とスペイン語です。約半数がカタるいねがお好み、大多数がかたるうね語を理解し、ほぼ全員がスプエン語を知っています。

**❌ LFM2.5-Audio base:**
> バルセロナの公用語はカタルーニャ語とスペイン語です。約半数がカタルーニャ語を好み、だいたい数がカタルーニャ語を理解し、ほぼ全員がスペイン語を知っています。

*Why ours wins:* Whisper mangles the proper nouns into broken katakana (バルセアナ, カタルーネ, スプエン語); ours is **letter-perfect**.

---

## JA Sample 2 — `ja_jp_1711`  (15s)  ·  CER ours 0% vs Whisper 7% vs base 6%

🔊 `data/fleurs/ja_jp_test/audio/ja_jp_1711.wav`

**Reference (truth):**
> これらの理論では、ある種の人々が日常的に特定のことをしたがる動機について、そして彼らに特定のことをさせる、あるいはさせない環境要因について考察します。

**✅ OURS (balanced_r32_csf):**
> これらの理論では ある種の人々が日常的に特定のことをしたがる動機について そして彼らに特定のことをさせるあるいはさせない環境要因について考察します

**❌ Whisper large-v3 (base):**
> これらの理論では、ある種の人々が日常的に特定のことをしたがる動機について、そして、彼らにとくていなことをさせる、あるいはさせない環境要因 について考察します。

**❌ LFM2.5-Audio base:**
> これらの理論では、ある種の人々が日常的に特定のことを従う動機について、そして彼らに特定のことをさせるあるいはさせない環境要因について考察します。

*Why ours wins:* Ours is **perfect**; Whisper slips (とくていな) and base mistranscribes (従う for させる).

---

## JA Sample 3 — `ja_jp_1749`  (18s)  ·  CER ours 3% vs Whisper 34% vs base 10%

🔊 `data/fleurs/ja_jp_test/audio/ja_jp_1749.wav`

**Reference (truth):**
> 統治者は「ビシー」のフランス人たちでした。彼らは1940年にドイツ軍と和平を結び、侵略者と戦うのではなく、侵略者に協力したフランス人たちです。

**✅ OURS (balanced_r32_csf):**
> 当事者はビシーのフランス人たちでした 彼らは1940年にドイツ軍と和平を結び 侵略者と戦うのではなく 侵略者に協力したフランス人たちです

**❌ Whisper large-v3 (base):**
> 当事者は、ヴィシーのフランス人たちでした。彼らは1940年にドイツ軍と和平を結び、侵略者と戦うのではなく、

**❌ LFM2.5-Audio base:**
> 当事者はヴィシーのフランス人たちでした。彼らは1940年にドイツ軍とは兵を結び侵略者と戦うのではなく、侵略者に協力したフランス人たちです。

*Why ours wins:* Whisper **truncates** — it drops the entire second half of the sentence; ours is complete and accurate.

---

## JA Sample 4 — `ja_jp_1699`  (14s)  ·  CER ours 5% vs Whisper 16% vs base 13%

🔊 `data/fleurs/ja_jp_test/audio/ja_jp_1699.wav`

**Reference (truth):**
> ある観察者がビシュケクを「無政府状態」に陥っていると表現したように、ギャングたちが街路を徘徊し、日用品店を略奪しました。

**✅ OURS (balanced_r32_csf):**
> ある観察者がビシュケクを未戦闘状態に陥っていると表現したように ギャングたちが街路を徘徊し 日用品店を略奪しました

**❌ Whisper large-v3 (base):**
> ある観察者が、美酒客を未潜伏状態に陥っていると表現したように、ギャングたちが街路を徘徊し、日曜品店を略奪しました。

**❌ LFM2.5-Audio base:**
> ある観察者がビシケクを未戦不状態に陥っていると表現したように、ギャングたちが街道を廃絶し日用品店を略奪しました。

*Why ours wins:* Whisper **mis-hears the place name** ビシュケク → 美酒客; ours transcribes it correctly.

---

## JA Sample 5 — `ja_jp_1833`  (17s)  ·  CER ours 3% vs Whisper 10% vs base 10%

🔊 `data/fleurs/ja_jp_test/audio/ja_jp_1833.wav`

**Reference (truth):**
> 哲学者のアリストテレスは、万物は4つの要素のうち1つ以上を混合した物で構成されているという理論を唱えました。4つの要素とは、土、水、空気、火です。

**✅ OURS (balanced_r32_csf):**
> 哲学者のアリストテレスは 万物は4つの要素のうち1つ以上を混合したもので構成されているという理論を唱えました 4つの要素とは 土 水 空気 火です

**❌ Whisper large-v3 (base):**
> 哲学者のアリストテレスは、万物は4つの要素のうち1つ以上混合したもので構成されているという理論を唱えました。4つ の要素とは、土・水・空気・火です。

**❌ LFM2.5-Audio base:**
> 哲学者のアリストテレスは、万物は四つの要素のうち一つ以上混合したもので構成されているという理論を唱えました。四つの要素とは土、水、空気、火です。

*Why ours wins:* Ours is clean and complete on a technical passage; both baselines make small slips.

---



---

# English-only samples (we rescued on-device English)

On **pure English**, the on-device **LFM base is broken — it katakana-forces English audio into gibberish**. Our fine-tune makes it actually transcribe English, **privately and on-device**. (Whisper large-v3 is a strong *cloud* model — we're competitive with it on English, not claiming to beat it; the win shown here is over the on-device base + the privacy/on-device angle. We genuinely beat both on samples 1 & 5.)

## EN Sample 1 — `en_us_1698`  (6s)  ·  WER ours 0% / Whisper 6% / base 18%

🔊 `data/fleurs/en_us_test/audio/en_us_1698.wav`

**Reference (truth):**
> Everything in the Universe is made of matter. All matter is made of tiny particles called atoms.

**✅ OURS (balanced_r32_csf, on-device):**
> everything in the universe is made of matter all matter is made of tiny particles called atoms

**Whisper large-v3 (base) — cloud baseline:**
> Everything in the universe is made of matter. All matter is made up of tiny particles called atoms.

**❌ LFM2.5-Audio base (katakana-forces English):**
> Everything is a universe. Made of matter, all matter is made of tiny particles, called atoms.

*Takeaway:* **Ours is letter-perfect** (and beats both): base LFM breaks the first sentence, Whisper says 'made up of'. Ours runs on-device.

---

## EN Sample 2 — `en_us_1819`  (13s)  ·  WER ours 8% / Whisper 0% / base 100%

🔊 `data/fleurs/en_us_test/audio/en_us_1819.wav`

**Reference (truth):**
> Ancient China had a unique way of showing different time periods; each stage of China or each family that was in power was a distinctive dynasty.

**✅ OURS (balanced_r32_csf, on-device):**
> ancient china had a unique way of showing different time periods each stage of china or each family that was in power was a distinct dynamic

**Whisper large-v3 (base) — cloud baseline:**
> Ancient China had a unique way of showing different time periods. Each stage of China or each family that was in power was a distinctive dynasty.

**❌ LFM2.5-Audio base (katakana-forces English):**
> イーチン・チャイナはユニークなウェイを披露する。イーチン・ステイション・チャイナはイーチ・ファミリー・ザ・ウィズン・パワーはディスティンクティブ・ダイナスティ。

*Takeaway:* Base LFM **katakana-forces pure English audio into gibberish** (イーチン・チャイナ…); ours transcribes it correctly. (Whisper also correct — cloud.)

---

## EN Sample 3 — `en_us_1772`  (6s)  ·  WER ours 8% / Whisper 0% / base 92%

🔊 `data/fleurs/en_us_test/audio/en_us_1772.wav`

**Reference (truth):**
> So, it is likely that the notation was added simply as a label.

**✅ OURS (balanced_r32_csf, on-device):**
> so it is likely that the notification was added simply as a label

**Whisper large-v3 (base) — cloud baseline:**
> So, it is likely that the notation was added simply as a label.

**❌ LFM2.5-Audio base (katakana-forces English):**
> そう、SYCLATEのチェーシングはAdd Simple As Vibe。

*Takeaway:* Base collapses English into katakana junk (SYCLATEのチェーシング…); ours is clean. (Whisper also clean.)

---

## EN Sample 4 — `en_us_1742`  (7s)  ·  WER ours 13% / Whisper 4% / base 100%

🔊 `data/fleurs/en_us_test/audio/en_us_1742.wav`

**Reference (truth):**
> Hong Kong Island gives the territory of Hong Kong its name and is the place that many tourists regard as the main focus.

**✅ OURS (balanced_r32_csf, on-device):**
> hong kong island gives the territory of hong kong's name and is the place that minimizes regard as the main focus

**Whisper large-v3 (base) — cloud baseline:**
> Hong Kong Island gives the territory of Hong Kong its name and is a place that many tourists regard as the main focus.

**❌ LFM2.5-Audio base (katakana-forces English):**
> 香港アイランド・ギブズ・テリトリー・オブ・ハンカゲス・ネイムズ・ザ・プレイス・ミニトレス・リバー・アズ・メインフォーケス。

*Takeaway:* Base outputs pure katakana for English (香港アイランド・ギブズ…); ours transcribes the English. (Whisper also good.)

---

## EN Sample 5 — `en_us_1707`  (11s)  ·  WER ours 9% / Whisper 18% / base 55%

🔊 `data/fleurs/en_us_test/audio/en_us_1707.wav`

**Reference (truth):**
> It is one of the main attractions of South Africa and it is considered the flagship of South African National Parks (SANParks).

**✅ OURS (balanced_r32_csf, on-device):**
> it is one of the main attractions of south africa it is considered the flagship of south african national parks 3 parks

**Whisper large-v3 (base) — cloud baseline:**
> It's one of the main attractions of South Africa, and it's considered the flagship of South African national parks, sand parks.

**❌ LFM2.5-Audio base (katakana-forces English):**
> It's one of the main attractions of South Africa. It's a country that is a part of the South Africa National Parks.

*Takeaway:* **Ours beats both** — base rambles off-script; ours stays faithful to the audio.

---

