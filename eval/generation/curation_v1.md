# Curation of the generation set (v1)

How 108 candidate questions became the 104 of `set_v1.yaml`.

1. **Independent review.** A separate agent that had not written the questions checked all 108 against fixed criteria: the evidence supports the whole expected answer; the question is unambiguous in the corpus; it reads like a fan and does not copy the evidence; a question that must decline really cannot be answered from the passages; follow-ups need their first turn; no near-duplicates. It searched the corpus for conflicts. Result: 81 kept as they were, 18 to fix, 3 to drop, 6 doubts.
2. **Verification.** The reviewer's corpus claims were checked against the chunks (9 checked, all correct), then the changes were applied. Evidence quotes were never edited; all 73 answerable quotes were re-checked word for word against their sections.
3. **Doubts.** Four of the six doubts (gen_071, 073, 086, 089) are superlative or count questions a passage could tempt a partial answer to ("Not Guilty had 102 takes"). They stay: declining an extreme that no passage states is exactly what they measure. gen_097 (DAW advice) stays for the same reason. gen_057 was dropped as a duplicate of gen_070.
4. **Approval.** The developer reviewed the 22 changed entries below and approved them on 2026-09-30; the set was then frozen as `set_v1.yaml`.

Kept 104: detail 38, unanswerable 19, credits 15, known 14, out_of_scope 10, follow_up 5, adversarial 3.

## Dropped (4)

- **gen_021** (detail): same passage as gen_020; near-verbatim a test pool question.
  - Q: What did Paul ask of the horn player on For No One that was unusual?
  - A: To play a note beyond the usual range of the instrument.
- **gen_038** (detail): most of the corpus says Martin hired Andy White; contradicts the expected answer.
  - Q: Who decided to bring in a session drummer for the first Beatles single?
  - A: Producer Ron Richards, who hired session drummer Andy White.
- **gen_057** (known): same fact as gen_070.
  - Q: Which member left the Beatles for two weeks during the White Album sessions?
  - A: Ringo Starr, in August 1968.
- **gen_076** (unanswerable): answerable by combining two passages, so declining is not the only right behaviour.
  - Q: Which Beatles album has the most George Harrison songs?
  - A: The sources here don't say.

## Fixed (18)

### gen_001 (detail)
Why: answer added "the rotating speaker", not in the evidence.

- question (unchanged): How did the engineers make John's voice sound like a far-off chanting lama on Tomorrow Never Knows?
- answer before: They fed part of his vocal through a Leslie speaker cabinet, the rotating speaker normally used for a Hammond organ.
- answer after: **They fed part of his vocal through a Leslie speaker cabinet, normally used for a Hammond organ.**
- evidence: Part of Lennon's vocal was fed through a Leslie speaker cabinet, normally used for a Hammond organ.

### gen_005 (detail)
Why: the question asked why; the evidence only says who set it off and where it rings.

- question before: Why is there an alarm clock ringing in the middle of A Day in the Life?
- question after: **Who set off the alarm clock you hear in the middle of A Day in the Life, and where does it ring?**
- answer before: Mal Evans set it off to mark the end of the 24-bar bridge.
- answer after: **Mal Evans; it rings at the end of the 24-bar bridge.**
- evidence: The 24-bar bridge ended with the sound of an alarm clock triggered by Evans.

### gen_010 (known)
Why: "opens" is not in the evidence, which says the recording "includes" feedback.

- question before: What unusual guitar sound opens I Feel Fine?
- question after: **What unusual guitar sound can you hear on I Feel Fine?**
- answer (unchanged): Guitar feedback, one of the earliest uses of it on a pop record.
- evidence: The recording includes one of the earliest uses of guitar feedback in popular music.

### gen_014 (detail)
Why: "three versions ... structure, key and tempo" is in the section, not in the evidence.

- question (unchanged): Why does the released Strawberry Fields Forever sound like two different songs stitched together?
- answer before: It combines the last two of three versions they recorded, which differed in structure, key and tempo.
- answer after: **It was made by combining the final two of the versions they recorded.**
- evidence: the released recording was created through a combination of the final two versions.

### gen_016 (credits)
Why: "on the Please Please Me album" is not in the evidence.

- question before: Who drums on the Love Me Do heard on the Please Please Me album?
- question after: **Ringo didn't drum on the remake of Love Me Do. Who did, and what did Ringo play instead?**
- answer (unchanged): Session drummer Andy White; Ringo was moved to tambourine.
- evidence: A week later, The Beatles recorded "Love Me Do" with session drummer Andy White on drums. Starr was relegated to playing tambourine.

### gen_028 (detail)
Why: the corpus also names tambura (wp_1451526__s1); the evidence says "such as".

- question (unchanged): Which Indian instruments can you hear on George's Love You To?
- answer before: Sitar and tabla.
- answer after: **Indian instruments such as sitar and tabla.**
- evidence: The song was written and sung by George Harrison and features Indian instrumentation such as sitar and tabla.

### gen_029 (detail)
Why: "as a lead line ... the first bass through a fuzzbox" is not in the evidence.

- question (unchanged): Why are there two bass parts on Think for Yourself?
- answer before: One is a standard bass and the other is played through a fuzzbox as a lead line, the first bass recorded through a fuzzbox.
- answer after: **One is a standard bass part and the other is played through a fuzzbox.**
- evidence: In a departure from convention, the track includes two bass guitar parts – one standard and one played through a fuzzbox.

### gen_032 (detail)
Why: "bounced tracks together" is not in the evidence.

- question (unchanged): Why was Sgt. Pepper made on four-track tape when eight-track already existed?
- answer before: Eight-track tape recorders were not working in commercial London studios until late 1967, so they bounced tracks together on four-track machines.
- answer after: **Because eight-track recorders were not working in commercial London studios until late 1967.**
- evidence: The album was recorded using four-track equipment, since eight-track tape recorders were not operational in commercial studios in London until late 1967.

### gen_033 (known)
Why: ambiguous: "because it is on Abbey Road" was a fair literal answer.

- question before: Why is the studio on Abbey Road called that?
- question after: **When and why did EMI's studio get renamed Abbey Road?**
- answer before: EMI's studio was renamed Abbey Road in 1976, after the Beatles' 1969 album.
- answer after: **In 1976, after the Beatles' 1969 album Abbey Road.**
- evidence: In 1976, the studio was renamed from EMI to Abbey Road after the band's 1969 album.

### gen_036 (credits)
Why: the corpus disagrees on the count: eight concerts (wp_869236__s0), five (wp_25832__s2).

- question (unchanged): Who drummed with the Beatles when Ringo missed part of the 1964 world tour?
- answer before: Jimmie Nicol, for eight concerts.
- answer after: **Jimmie Nicol.**
- evidence: He is best known for sitting in for Ringo Starr in the Beatles for eight concerts of the Beatles' 1964 world tour during the height of Beatlemania.

### gen_039 (detail)
Why: "used for Abbey Road" could mean the studio, which used the REDD console (wp_140630__s2).

- question before: Which classic EMI mixing desk was used for Abbey Road?
- question after: **Which EMI console was the Beatles' Abbey Road album mixed on?**
- answer before: The EMI TG12345, designed by EMI for its Abbey Road Studios.
- answer after: **The EMI TG12345.**
- evidence: The EMI TG12345 was a mixing console designed by EMI for their Abbey Road Studios, which was used to mix several influential albums, including the Beatles' Abbey Road and Pink Floyd's The Dark Side of the Moon.

### gen_046 (credits)
Why: the drums on the released song are a composite of three Beatles (wp_75350__s4).

- question before: Who played drums on Back in the U.S.S.R.?
- question after: **Who played what on the basic track of Back in the U.S.S.R.?**
- answer before: Paul McCartney, with George on electric guitar and John on a Fender Bass VI.
- answer after: **Paul on drums, George on electric guitar and John on a Fender Bass VI.**
- evidence: Five takes were recorded of the basic track, featuring McCartney on drums, George Harrison on electric guitar, and John Lennon on Fender Bass VI.

### gen_048 (credits)
Why: the question copied "tenor saxophone solo" from the evidence.

- question before: Who played the tenor saxophone solo on Lady Madonna?
- question after: **Who's the sax player on Lady Madonna?**
- answer (unchanged): British jazz musician and club owner Ronnie Scott.
- evidence: The tenor saxophone solo on the track was played by British jazz musician and club owner Ronnie Scott.

### gen_054 (known)
Why: the album also includes earlier single sides, so "one day" did not cover it all.

- question before: How long did it take to record the Please Please Me album?
- question after: **How long was the main recording session for the Please Please Me album?**
- answer (unchanged): One day, 11 February 1963, with a few overdubs added later.
- evidence: The Beatles recorded Please Please Me in one day at EMI Studios on 11 February 1963, with Martin adding overdubs to "Misery" and "Baby It's You" nine days later.

### gen_056 (known)
Why: neither the "Beatle Bass" nickname nor the "because" is in the evidence.

- question before: Why is the Höfner violin bass called the Beatle Bass?
- question after: **Which Beatle made the Höfner violin bass famous?**
- answer before: Because it became famous in the 1960s as the main bass of Paul McCartney.
- answer after: **Paul McCartney; it was his main bass in the 1960s.**
- evidence: It was introduced in 1955 and gained fame during the 1960s as the primary bass used by Paul McCartney of the Beatles.

### gen_061 (detail)
Why: "side two" is not in the evidence ("the remainder of the album").

- question before: Why does side two of the Yellow Submarine album contain no Beatles songs?
- question after: **Apart from the Beatles songs, what's on the Yellow Submarine album?**
- answer before: It is George Martin's re-recording of music from the film's orchestral soundtrack.
- answer after: **George Martin's re-recording of selections from the film's orchestral soundtrack.**
- evidence: The remainder of the album is a re-recording of selections from the film's orchestral soundtrack by the band's producer, George Martin.

### gen_062 (detail)
Why: "a year of world tours" goes beyond the evidence.

- question (unchanged): Why does Beatles for Sale sound less upbeat than the earlier albums?
- answer before: Partly because the band was exhausted after a year of world tours.
- answer after: **Partly because the band was exhausted after a series of tours in 1964.**
- evidence: The album marked a departure from the upbeat tone that had characterised the Beatles' previous work, partly due to the band's exhaustion after a series of tours that had established them as a worldwide phenomenon in 1964.

### gen_063 (detail)
Why: "Cry for a Shadow" (1961, co-written with Lennon) came earlier (wp_12731__s2).

- question before: What was George Harrison's first song recorded by the Beatles?
- question after: **What was George Harrison's first solo composition recorded by the Beatles?**
- answer (unchanged): Don't Bother Me, on With the Beatles.
- evidence: Produced by George Martin, the album features eight original compositions (seven by Lennon–McCartney and "Don't Bother Me", George Harrison's first recorded solo composition)
