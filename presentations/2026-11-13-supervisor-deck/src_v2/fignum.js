// Slide text that depends on the figures (numbers from the figure agents' JSON / reports and computed.json).
const path = require('path');
const { N, CP, b, t, f, mean } = require('./numbers.js');
const F9 = require('./F9_numbers.json');
const FN = {};
const T = (s) => t(s);

// ── real-data slides
FN.strongBullets = [
  [b('a. Muscle signals: '), T('the calf switches on when YES pushes. When everything is low, YES is relaxed (green).')],
  [b('b. Measured torque: '), T('even while relaxed, the sensor reads about −4 Nm here. That is the passive torque at this angle (toes up, so the calf is stretched and pulls back).')],
  [b('During a push '), T('the reading goes to about −32 Nm.')],
  [b('c. Voluntary torque '), T('= −32 − (−4) ≈ −28 Nm made by the muscles, and 0 between pushes. This is what the model has to predict.')],
  [b('Here the passive value is easy: '), T('an error of 1–2 Nm in it hardly matters next to a 28 Nm push.')],
];
FN.strongNotes = 'Real data from YES, session 2, foot locked toes up. Top: the muscle signals; the calf switches on with every push. Middle: the measured torque. Even when YES is relaxed, the sensor reads about minus 4 Nm: that is the passive torque. During a push it goes to about minus 32. Bottom: subtract the passive line and you get the voluntary torque, about minus 28 Nm per push and 0 when relaxed. At this angle a small mistake in the passive value does not matter, because the pushes are huge.';
FN.toesBullets = [
  [b('Small pushes: '), T('at −0.49 rad the calf is short and YES can only push about 1 Nm (typical push 0.9 Nm).')],
  [b('Relaxed level inside this trial: '), T('about 9 Nm at the start, about 8.4 Nm at the end (it slides down by about 0.5 Nm).')],
  [b('Old method: '), T('subtracted 10.6 Nm, a value from a separate relaxed recording made before the trial. That is 2 Nm too much, twice the size of a push.')],
  [b('So: '), T('the old “voluntary torque” (blue, panel c) sits at about −2 Nm even when YES is relaxed: a push that is not there.')],
  [b('The model was trained and graded '), T('with this wrong answer. This offset is the main reason the angle fails: R² −1.1 as scored last semester, −14 against the shared answer key (introduced later).')],
];
FN.toesNotes = 'Same thing at the most toes-down angle. Here YES can only push about 1 Newton-metre. The old method subtracted 10.6 Nm, taken from a separate relaxed recording, but the real relaxed level inside this trial is about 9 going down to 8.4. So the old voluntary torque was 2 Nm too low the whole time, twice the size of the pushes. The model learned that wrong answer and was graded on it. That offset is the main reason for the R² of minus 14 at this angle.';

// ── old table
FN.tableBullets = [
  [b('What the table was: '), T('one passive torque value per angle. It turned measured torque into the “right answer” (voluntary = measured − table value) for training and for grading.')],
  [b('Data it came from: '), T('before each pushing trial there is a separate relaxed recording at the same angle. Its first 20 s (30 s for YES) are a clean relaxed hold (foot still, torque steady). After that the angle signal jumps and the torque signal swings far beyond anything physical; what the device does then is not documented.')],
  [b('How it was computed: '), T('cut each relaxed recording where the angle signal jumps; keep every piece whose torque spread is under 20 Nm and whose average is within ±20 Nm; average each kept piece → one point; merge points closer than 0.05 rad; draw straight lines between the points.')],
  [b('How it was used: '), T('read the line at the pushing trial’s angle and subtract that one number from the whole 90 s trial.')],
  [b('YES: '), T('typed in from the lab worksheet (they match the clean holds in the file). '), b('IES: '), T('no relaxed recordings, so nothing was subtracted.')],
];
FN.tableNotes = 'This is how the old passive table worked. Before each pushing trial there is a separate relaxed recording at the same angle; only its first 20 seconds are a clean relaxed hold. The code cut these recordings into pieces, kept every piece whose torque did not vary by more than 20 Nm, averaged each piece, and drew a curve through the points: one passive value per angle. That one number was subtracted from the whole pushing trial. The problem on the left: for JM, pieces that were clearly not relaxed passed the 20 Nm rule and created a fake point of plus 11.7 Nm at minus 0.07 rad, as the bottom graph shows.';
FN.wrongBullets = [
  [b('Check: '), T('we compared each table value with the relaxed level measured inside the pushing trial itself (the relaxed moments between pushes). 0 = the table was right.')],
  [b('Even clean holds are off: '), T('the relaxed torque inside a pushing trial is only about 80 % of the separate hold. Typical difference 0.6 Nm; about 1 in 5 trials off by more than 1 Nm.')],
  [b('Toes-down angles: '), T('the table was too high in 20 of 20 trials (up to 2 Nm for YES), exactly where the pushes are only about 1 Nm.')],
  [b('How we found it: '), T('at JM −0.08 rad the relaxed parts of the “right answer” sat at about −13 Nm instead of 0. Last semester’s score hid it (R² 0.45), because the answer itself was shifted; against the shared key it is −60. That led back to the table entry. Then we did this comparison for everyone.')],
];
FN.wrongNotes = 'To check the old values we compared them with the relaxed level measured inside each pushing trial. Zero means the table was right. JM’s bug is off the scale: 12 to 14 Nm. But even without the bug, the separate relaxed recording does not match the trial: inside a pushing trial the relaxed torque is only about 80 percent of the separate hold, and at the toes-down angles the table was too high in all 20 cases. The literature explains why: passive torque relaxes over time and gets looser after contractions.';
FN.driftBullets = [
  [b('What: '), T('tissue held stretched slowly relaxes (“stress relaxation”), so the relaxed torque slides during the 90 s. Pushing also “loosens” it a little.')],
  [b('How we saw it: '), T('the torque in each relaxed stretch (green dots) is not flat across the trial.')],
  [b('How big: '), T('usually small: median change 0.15 Nm from the first to the last 15 s; in 72 % of trials it moves toward zero.')],
  [b('Why it matters at toes down: '), T('YES at −0.49 rad drops 0.5 Nm while its pushes are only about 1 Nm. One number per trial (old method) cannot follow that.')],
  [b('Literature: '), T('a relaxed ankle held stretched loses up to ~20 % of its torque within 60 s (McNair 2001) and keeps relaxing for minutes (Duong 2001); after calf contractions passive torque is about 5 % lower (Kay & Blazevich 2009).')],
];
FN.driftNotes = 'Passive drift: even inside one 90-second trial the relaxed torque is not constant. Tissue that is held stretched slowly relaxes; this is well known as stress relaxation. We saw it by plotting the torque in each relaxed stretch over the trial. Usually the change is small, about 0.15 Nm, but at YES’s most toes-down angle it is 0.5 Nm while the pushes are only about 1 Nm. A single number per trial cannot follow that; a line through the relaxed moments can.';

FN.jmBullets = [
  [b('Blue curve = the old table: '), T('each dot is the average of one kept piece; the old method read this curve at each pushing trial’s angle (black ticks = the 8 trial angles).')],
  [b('The bad entry: '), T('+11.7 Nm (session 1) and +10.75 Nm (session 2) near −0.07 rad, from pieces that were not relaxed. Its neighbours are −0.5 and −3.4 Nm.')],
  [b('Effect: '), T('for JM’s trial at −0.08 rad the old method subtracted about 11 Nm, while the real relaxed level was about −2.4 Nm: 13.7 Nm too much (12.0 Nm in session 2).')],
  [b('So '), T('even when JM was relaxed, the old “voluntary torque” sat at about −13 Nm, and the model learned a push that never happened. Its score looked acceptable (R² 0.45) only because it was graded with the same wrong answer.')],
  [b('Orange dots: '), T('the relaxed level measured inside each pushing trial. Away from the bug they are close to the curve but mostly a bit below it (next slide).')],
];
FN.jmNotes = 'Here is the table that came out for JM. Each blue dot is one averaged piece and the line is what the old method read off for each pushing trial. One point is clearly wrong: plus 11.7 Nm where its neighbours are around minus 0.5 and minus 3.4. It came from pieces of the relaxed recording that were not relaxed at all. For JM’s trial at minus 0.08 rad that meant subtracting almost 14 Nm too much, so the training answer showed a big push while JM was relaxed. The orange dots are the relaxed level measured inside the pushing trials, the new method.';

// ── EMG
FN.emgWhyBullets = [
  'The model learns “this much EMG = this much torque” from one session.',
  'If the other session gives a smaller EMG for the same push, the model predicts a smaller push (and the other way round).',
  'The change differs per person, per angle and per muscle: from 0.4× to 2.4× (calf average). JM at −0.64 rad: about half.',
];
FN.emgCauseBullets = [
  'Electrode–skin contact changes over time (Hewson 2003); more sweat under the electrodes → smaller EMG (Abdoli-Eramaki 2012).',
  'Skin temperature (cooler skin → soleus EMG about 2×: Winkel & Jørgensen 1991).',
  'Fatigue: EMG size for the same force can go up or down (Dimitrova & Dimitrov 2003).',
  'The person shares the push differently between muscles (EG −0.74 rad: GM 3.4× bigger, SOL 0.45×), and small shifts of muscle under the electrode (De Luca 1997). We cannot tell which from our data.',
];
FN.emgNotes = 'Problem four: for the same push, the EMG comes out a different size in session 2. It goes both ways and differs per angle and per muscle, from about 0.4 to 2.4 times. Even with the electrodes left on, the electrode-skin contact changes with sweat, temperature changes the signal, fatigue changes it, and people can share the push differently between muscles. We cannot tell which one it was. It matters because the model learned the link between EMG size and torque in one session.';
FN.agcBullets = [
  [b('Idea: '), T('like the automatic volume control in a radio or hearing aid (Wheeler 1928; Moore 1991): keep each muscle signal at a steady scale, whatever its raw size.')],
  [b('How, per muscle, every 0.5 s: '), T('“floor” = its quiet level in the past 20 s (10th percentile); “top” = its strong-push level in the past 45 s (95th percentile). Output = (signal − floor) ÷ (top − floor): 0 = relaxed, 1 = a recent strong push.')],
  [b('If session 2 is half the size, '), T('floor and top are half too, so the output is about the same. JM −0.64 rad: the push-related EMG rise was 0.40× of session 1 before, about 0.75–0.8× after (most of the gap, not all).')],
  [b('Only past EMG, never torque: '), T('it could run live. At the start of a trial it uses values from the learning session.')],
  [b('Off for CH: '), T('if training trials push both ways (some > +2 Nm, others < −2 Nm), rescaling each muscle hides which muscle is working. CH’s flipped angle: R² −3.3 / −5.5 with it, 0.98 / 0.95 without.')],
  [b('Literature: '), T('EMG must be normalised to compare sessions (Burden 2010; Besomi 2020). Running rescaling has been used before (Koller 2015; Tanaka 2022), and high percentiles instead of the max are common (Jonsson 1982). Our exact running-percentile version is our own design.')],
];
FN.agcNotes = 'The fix is auto-gain, an automatic volume control for each muscle, the same idea as in a radio or a hearing aid. For each muscle we keep track of its recent quiet level and its recent strong-push level, using only the past, and rescale the signal between them. If the signal is half the size in session 2, both levels are half the size too, so the output is about the same. For JM it brings the session-2 signal from 0.4 to about 0.75 of session 1. It is switched off for people who push in both directions, which is only CH. Taking it out of the new pipeline lowers the main three from 0.93 to 0.88, mostly because of JM and EG; for YES and HM it is slightly better without it.';

// ── fit / scaling
FN.lossBullets = [
  [b('Fitting = least squares: '), T('the computer adjusts the model to make the sum of squared errors over all training samples as small as possible, all 8 angles together.')],
  [b('The problem: '), T('errors are in Nm. An angle where the torque swings by 15 Nm produces squared errors hundreds of times bigger than an angle where it swings by 0.6 Nm. For YES the 2 most toes-down angles carry 1 % of the total, the 3 toes-up angles 79 %.')],
  [b('Result: '), T('the fit is happy to be wrong at the toes-down angles. YES is the extreme case (557× between its biggest and smallest angle; the other people 18–49×).')],
];
FN.lossNotes = 'Problem five is about the fit itself. Fitting the network is least squares over all angles together. Because the errors are in Newton-metres, the strong angles, where the torque swings by 15 Nm, completely dominate the sum. For YES the two most toes-down angles carry about 1 percent of the total, so the fit hardly cares about them.';
FN.scaleBullets = [
  [b('Typical swing (σ): '), T('the standard deviation of the voluntary torque at one angle in the training data (YES: 0.64 Nm at −0.49 rad, 15.0 Nm at +0.14 rad).')],
  [b('What we do: '), T('divide each angle’s target by its σ before fitting, and multiply the prediction back by σ afterwards, so the output is in Nm again.')],
  [b('Same as weighted least squares '), T('with weight 1/σ² per angle: every angle gets the same say (12.5 % each). Networks that learn several things at once use a similar 1/σ² weighting (Kendall et al. 2018; there σ is fitted too).')],
  [b('What it achieves: '), T('PLACEHOLDER_SCALE')],
  [b('Extra weight on top did not help: '), T('×3 more on the toes-down angles tied, ×10 was worse (backup slide).')],
];
FN.scaleNotes = 'The fix: we divide each angle’s torque by its typical swing, its standard deviation, before fitting, and multiply the prediction back afterwards. For a modeller this is simply weighted least squares with weight one over sigma squared per angle: now every angle has the same say in the fit. When we remove only this step from the new pipeline, the toes-down angles get clearly worse.';

// ── model
FN.lstmBullets = [
  'Most of the improvement came from the data, not the network: the old small network inside the new pipeline reaches 0.921 vs 0.930 on the main 3 (1 run each).',
  'In our first report this semester we tried other sizes and common network designs on the same fixed data (a bigger LSTM with 32 cells, GRU, CNN, a plain network). None was better everywhere; some were much worse at single angles.',
  '16 cells + the per-angle corrections gave a small but consistent gain: +0.008 average R², better on HM (0.93 vs 0.90), and no more overshoot spikes (the small one sometimes predicted −4.5 Nm where the truth was −2 Nm).',
  'Honestly, 16 was not picked by a full search: we doubled 8 to give the per-angle corrections more to work with, it helped a little, and it is still tiny (3,665 numbers vs ~57,000 training windows; about 5 minutes to fit on a normal PC).',
];
FN.lstmNotes = 'We kept the same type of network, an LSTM, which reads the 0.2-second window step by step and keeps a small memory. Why 16 memory cells instead of 8? Honestly, we did not do a full search. Last semester we tried bigger networks and other types on the same data and none was better everywhere. Going from 8 to 16, together with the per-angle corrections, gave a small but consistent gain and removed some overshoot spikes. Most of the improvement comes from the data steps: the old small network with the new data steps is almost as good.';
FN.expertsAchieve = 'a small but consistent gain (+0.008 average R², HM 0.93 vs 0.90) and no more overshoot spikes. Most of the improvement comes from the data steps, not from this (Jacobs et al. 1991 for the general idea).';

// ── checking
FN.keyBullets = [
  [b('Before: '), T('each pipeline was graded against its own “right answer” (measured − its own passive value).')],
  [b('The trap: '), T('if the passive value is wrong, the training answer and the grading answer are wrong in the same way. The model learns the wrong level and is graded on the same wrong level, so it looks OK.')],
  [b('Example: '), T('YES −0.49 rad, same old prediction: R² −1.1 (error 0.57 Nm) against its own answer, R² −14.5 (error 1.53 Nm) against the shared key.')],
  [b('Now: '), T('one answer key for every model = measured torque − passive level measured inside that same test trial. The models never see it; it is only used for grading.')],
  [b('Not rigged for the new method: '), T('the version from our first report, whose answers are built differently (separate recordings + a correction), scores the same on this key (0.925 vs 0.925).')],
];
FN.keyNotes = 'Last semester each pipeline was graded against its own right answer. The trap is that a wrong passive value makes the training answer and the grading answer wrong in exactly the same way, so the model looks fine. Here the same old prediction gets R² minus 1.1 against its own answer but minus 14.5 against the shared answer key. Now every model, old and new, is graded against one answer key: measured torque minus the passive level measured inside the test trial. The models never see it.';
FN.phantomBullets = [
  'Phantom torque = the average size of the torque the model predicts while the person is relaxed (relaxed moments found from the EMG). It should be 0. R² hides it partly, so we report it separately.',
  'Old: 1.25 Nm on the main 3 (JM 2.3 Nm) and 1.07 Nm on the extra 3. New: 0.17 Nm on both.',
];
FN.phantomNotes = 'Phantom torque is the torque the model predicts while the person is relaxed. It should be zero. On the left the old model sits about 1 Nm below zero during every relaxed moment; the new one sits at zero. On the right, per person: the old pipeline had 0.5 to 2.3 Nm of phantom torque, the new one 0.13 to 0.24.';
FN.runsBullets = [
  'Fitting a network starts from random numbers and goes through the data in a random order, so two runs on the same data give slightly different models (like a nonlinear fit started from different initial guesses).',
  'We do 3 runs and report the average, never the best one. Old runs could disagree badly (left: R² 0.40, −0.03, −12.8); the new runs agree (0.92, 0.92, 0.92).',
];
FN.runsNotes = 'Training a network is a fit that starts from random numbers, a bit like a nonlinear fit started from different initial guesses. So we repeat it three times and average; we never pick the best run. The old pipeline could give very different answers from run to run, as on the left; the new one gives almost the same answer every time.';

// ── results
const p = (x) => f(x);
FN.summaryBullets = [
  [b('Main 3 (3 runs each): '), T(`average R² ${f(N.mainOld.r2, 3)} → ${f(N.mainNew.r2, 3)}; toes-down R² ${p(N.mainOld.pf2)} → ${p(N.mainNew.pf2)}; toes-down error ${p(N.mainOld.err)} → ${p(N.mainNew.err)} Nm.  `),
   b('Extra 3 (1 run each): '), T(`average R² ${f(N.extraOld.r2, 3)} → ${f(N.extraNew.r2, 3)}; toes-down R² ${p(N.extraOld.pf2)} → ${p(N.extraNew.pf2)}; toes-down error ${p(N.extraOld.err)} → ${p(N.extraNew.err)} Nm.`)],
  [b('Per angle: '), T('new beats old at 88 of the 96 single-angle results (6 people × 2 directions × 8 angles). The 8 others are small drops at middle or toes-up angles (5 of them HM), never at the toes-down angles. Main 3 at the toes-down angles: new better in 12 of 12 (Wilcoxon signed-rank test p = 0.0005; but these are 12 results from only 3 people, so not fully independent).')],
];
FN.summaryNotes = 'Here is every person, old versus new, both directions averaged, graded with the same answer key. On the left the average over all angles, in the middle the two most toes-down angles, on the right the error in Newton-metres at those angles. Every person improves. For YES and JM the toes-down R² goes from negative to 0.7 and 0.9. The extra three people, checked with one run each, reach 0.97 at the toes-down angles. New beats old at 88 of the 96 single-angle results; the few exceptions are small and never at the toes-down angles.';
FN.errorNotes = 'The same comparison in Newton-metres. Lower is better. The error is naturally bigger at the toes-up angles because the pushes are bigger there. JM’s old error at minus 0.08 rad is off the scale because of the table bug.';
FN.susBullets = [
  [b('R² is not “% accuracy”: '), T('it compares the error with how much the torque moves. At strong angles the torque swings by ±15–30 Nm, so an error of 1–2 Nm still gives 0.98–0.99.')],
  [b('47 of 96 trials have R² ≥ 0.98: '), T('their error is still 0.3–3.0 Nm (median 1.05 Nm). Only 6 of them are toes-down angles.')],
  [b('Last semester saw it too: '), T('at those same trials the old pipeline scored a median R² of 0.975 against its own answer.')],
  [b('Pushing parts only: '), T('average R² 0.89 (0.925 with the relaxed parts), so it is not the relaxed zeros that make it high.')],
  [b('Even a straight-line model '), T('gets 0.87–0.975 at those strong angles: the task is easy there.')],
  [b('No leak: '), T('the tested session is never used for fitting or for deciding when to stop; the answer key is never an input; auto-gain only uses past EMG. An independent code review checked this.')],
];
FN.susBox = [
  b('Compared with published work: '), T('isometric EMG → torque errors of about 4–5 % of max effort (elbow; Clancy 2012, Liu 2015); ours is about 3 % on the main 3. Same-session R² of 0.96–0.97 has been reported (wrist), dropping to 0.46–0.59 the next day (Ziai & Menon 2011): our two sessions are the same visit, so a different-day test is still missing.'),
];
FN.susNotes = 'You might think 0.99 is suspicious; we thought so too. R² is not accuracy: it compares the error with how much the torque moves. At the strong angles the torque swings by 15 to 30 Nm, so an error of one or two Newton-metres still gives 0.98 or 0.99. Last semester’s pipeline already had 0.97 to 0.99 at those angles. Without the relaxed parts it is still 0.89. Even a straight-line model gets up to 0.97 there. And we checked for leaks: the tested session is never used for fitting. What would be suspicious is a high score everywhere; instead the hardest angles are still the toes-down ones, and the error grows with the size of the push. Published isometric studies report similar errors, about 4 to 5 percent of max effort; ours is about 3 percent. But published work also shows that a different day is harder, and we have not tested that yet.';

// ── inspection notes from F9 numbers
const SUBJ = ['YES', 'JM', 'EG', 'IES', 'CH', 'HM'];
FN.inspectFoot = {};
FN.inspectNotes = {};
SUBJ.forEach(sj => {
  const o = F9.per_image[`${sj}_S1toS2_old`], n = F9.per_image[`${sj}_S1toS2_new`];
  const better = o.filter((x, i) => n[i].r2 > x.r2).length;
  const td = [0, 1].map(i => `${o[i].angle_label} rad: R² ${f(o[i].r2)} → ${f(n[i].r2)}, error ${o[i].rmse.toFixed(2)} → ${n[i].rmse.toFixed(2)} Nm`).join('; ');
  FN.inspectFoot[sj] = `Each panel = one angle, most toes-down top-left. Black = shared answer key, coloured = prediction (one run, whole 90 s trial); same y-scale for old and new. Toes-down: ${td}.`;
  FN.inspectNotes[sj] = `${sj}, learned from session 1 and tested on session 2, one training run. Each small panel is one angle, the most toes-down one top left. Black is the answer key, blue or orange the prediction. New is better at ${better} of 8 angles. At the two most toes-down angles: ${td}.` + (sj === 'CH' ? ' CH pulls the toes up at its most toes-down angles, so the torque there is positive.' : '') + (sj === 'IES' ? ' IES has no separate relaxed recordings, so the old pipeline subtracted nothing.' : '');
});
const H = F9.hardest;
FN.hardestNotes1 = 'The most toes-down angle of each main person, learned from session 1 and tested on session 2: the model was fitted on session 1 and then predicted session 2, which it never saw. Black is the answer key, blue the old pipeline, orange the new one. The old one gets the timing right but sits at the wrong level and often has the wrong size; the new one follows the pushes. YES is still the weakest: it only pushes about 1 Nm there, so an error of 0.3 Nm still limits R².';
FN.hardestNotes2 = 'The same for the three extra people. IES and CH were already fairly good with the old pipeline at this angle; HM improves from 0.08 to 0.91. CH pulls the toes up at this angle, so its torque is positive.';

// ── sources (verified by the literature agents; DOI or PubMed in the Q&A file)
const S = (cite, what) => [b(cite + ' — '), T(what)];
FN.sources = [
  [
    [b('Passive torque: why we measure it inside the trial', { color: 'EB6834', bullet: false })],
    S('Duong et al. 2001, Clin Biomech 16(7)', 'relaxed ankle held stretched: passive torque keeps falling for minutes (stress relaxation).'),
    S('McNair et al. 2001, Med Sci Sports Exerc 33(3)', 'calf held stretched: force drops by up to ~20 % within 60 s, mostly in the first 20 s.'),
    S('Magnusson et al. 1996, Scand J Med Sci Sports 6(6)', '90 s static stretch: passive torque falls by about a third with no muscle activity.'),
    S('Kay & Blazevich 2009, J Appl Physiol 107(4)', 'after calf contractions, passive ankle torque is about 5 % lower.'),
    S('Lakie & Campbell 2019, J Appl Physiol 126(6)', 'review: a relaxed muscle’s stiffness depends on what it just did (thixotropy).'),
    S('Hoang et al. 2005, J Biomech 38(6); Walker et al. 2022, Scand J Med Sci Sports 32(1)', 'passive ankle measures differ between sessions.'),
    S('Anderson et al. 2010, J Biomech 43(6)', 'a dynamometer’s relaxed reading also includes the weight of the foot and footplate.'),
    [b('EMG size: why auto-gain', { color: 'EB6834', bullet: false })],
    S('De Luca 1997, J Appl Biomech 13(2); Vigotsky et al. 2018, Front Physiol 8:985', 'electrode position, sweat and fat change the EMG size, not just effort.'),
    S('Hewson et al. 2003, J Electromyogr Kinesiol 13(3)', 'electrodes left on for 2 h: the electrode–skin contact (impedance) changed, for one of two electrode types.'),
    S('Abdoli-Eramaki et al. 2012, J Electromyogr Kinesiol 22(6)', 'more sweat under the electrodes → smaller EMG.'),
    S('Winkel & Jørgensen 1991, Eur J Appl Physiol 63(5)', 'cooler skin: soleus EMG about doubled for the same exercise.'),
    S('Dimitrova & Dimitrov 2003, J Electromyogr Kinesiol 13(1)', 'with fatigue, EMG size for the same force can go up, stay or go down.'),
  ],
  [
    S('Burden 2010, J Electromyogr Kinesiol 20(6); Besomi et al. 2020, J Electromyogr Kinesiol 53', 'EMG amplitude must be normalised to compare sessions.'),
    S('Schulte et al. 2022, Front Robot AI 9:869476', 'EMG → knee torque network: about 6 % error the same day, 10–11 % on later days.'),
    S('Tkach et al. 2010, J NeuroEng Rehabil 7:21', 'changes in the EMG over time lower the accuracy of EMG models.'),
    S('Koller et al. 2015, J NeuroEng Rehabil 12:97; Tanaka et al. 2022, Sensors 22(13)', 'running / sliding-window rescaling of EMG used before.'),
    S('Jonsson 1982, J Hum Ergol 11(1)', 'use a high percentile of the EMG instead of the maximum, which is unreliable.'),
    S('Wheeler 1928, Proc IRE 16(1); Moore et al. 1991, Br J Audiol 25(3)', 'automatic gain control in radios and hearing aids.'),
    [b('Model, fitting and scoring', { color: 'EB6834', bullet: false })],
    S('Hochreiter & Schmidhuber 1997, Neural Comput 9(8)', 'the LSTM network.'),
    S('Jacobs et al. 1991, Neural Comput 3(1)', 'mixture of experts: small expert models, a gate picks or blends them.'),
    S('Kendall et al. 2018, CVPR', 'weight each task’s squared error by 1/σ² so big-spread tasks do not dominate.'),
    S('Prechelt 1998, Neural Netw 11(4); Srivastava et al. 2014, JMLR 15', 'early stopping and dropout to avoid over-fitting.'),
    S('Gupta et al. 2009, J Hydrol 377; Schaefli & Gupta 2007, Hydrol Process 21', 'R²-type scores are higher when the signal swings more.'),
    S('Clancy et al. 2012, IEEE TBME 59(1); Liu et al. 2015, IEEE TNSRE 23(6)', 'isometric EMG → torque: error about 4–5 % of max effort.'),
    S('Ziai & Menon 2011, J NeuroEng Rehabil 8:56', 'wrist torque R² 0.96–0.97 in the same session, 0.53–0.63 one hour later, 0.46–0.59 the next day.'),
    S('Kizyte et al. 2023, IEEE TNSRE 31', 'ankle torque from EMG: error 4–6 % of the torque range within a session.'),
  ],
];

// ── ablation (seed 0, all 6 people, from computed.json)
const ab = CP.ablation;
const MAIN = ['YES', 'JM', 'EG'], EXTRA = ['IES', 'CH', 'HM'];
const agg = (name, subs, key) => {
  const rows = Object.entries(ab[name] || {}).filter(([k]) => subs.includes(k.split('|')[0])).map(([, v]) => v[key]);
  return rows.length === subs.length * 2 ? mean(rows) : null;
};
const ABL = [
  ['new', 'New pipeline (everything)', true],
  ['E34_final_oldpassive', 'New, but with the old passive table'],
  ['E32_restanchor_noagc_moe', 'New, but without auto-gain'],
  ['E33_final_noscale', 'New, but without dividing by the typical swing'],
  ['E35_final_lstm8', 'New, but with the old small network (no per-angle corrections)'],
  ['old', 'Old pipeline (nothing changed)', true],
];
const AH = { bold: true, color: 'FFFFFF', fill: { color: '1F2A44' }, align: 'center', valign: 'middle', fontSize: 12 };
const hdr = [{ text: 'Version (1 training run each, both directions)', options: Object.assign({}, AH, { align: 'left', rowspan: 2 }) },
  { text: 'Main 3 (YES, JM, EG)', options: Object.assign({ colspan: 3 }, AH) }, { text: 'Extra 3 (IES, CH, HM)', options: Object.assign({ colspan: 3 }, AH) }];
const hdr2 = ['average R²', 'toes-down R²', 'toes-down error (Nm)', 'average R²', 'toes-down R²', 'toes-down error (Nm)']
  .map(x => ({ text: x, options: Object.assign({}, AH, { fill: { color: '52514E' }, fontSize: 11 }) }));
FN.ablation = {
  ablationRows: [hdr, hdr2].concat(ABL.map(([k, lab, bold]) => {
    const vals = [agg(k, MAIN, 'r2_all'), agg(k, MAIN, 'r2_pf2'), agg(k, MAIN, 'err_pf2'), agg(k, EXTRA, 'r2_all'), agg(k, EXTRA, 'r2_pf2'), agg(k, EXTRA, 'err_pf2')];
    const fill = k === 'new' ? 'FCE6DB' : k === 'old' ? 'E3EDFB' : 'FFFFFF';
    return [{ text: lab, options: { bold: !!bold, color: '1F2A44', fill: { color: fill } } }].concat(vals.map((v, j) => ({
      text: v === null ? 'running' : (j % 3 === 2 ? v.toFixed(2) : f(v, j % 3 === 0 ? 3 : 2)),
      options: { align: 'center', fill: { color: fill }, bold: !!bold, color: v !== null && v < 0 ? 'C0392B' : '1B1B1B' } })));
  })),
  ablationColW: [4.55, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3],
  ablationNoteY: 4.72,
};
FN.ablation.vals = (k, subs, key) => agg(k, subs, key);
module.exports = FN;
