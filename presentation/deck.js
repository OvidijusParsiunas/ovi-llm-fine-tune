// Minimal slide engine.
// Next: → ↓ Space PageDown, click.  Previous: ← ↑ PageUp.  Home/End jump.  F toggles fullscreen.
// The URL hash (#3, or #3.2 for slide 3 at step 2) remembers where you are, so a refresh keeps your place.
// Elements with class="fragment" appear one at a time as you press "next" on that slide.
// A slide with data-steps="N" has N build steps instead: at step k it gets the classes s1…sk,
// so CSS can change anything on the slide per step (e.g. `.s3 .cake { opacity: 1 }`).
// Add ?all to the URL to show every slide at its last step (handy for previewing/printing).
// A slide with data-intro="<kind>" plays an entrance when reached by "next": it gets the class "intro",
// the slide you came from stays visible with the class "outro", and the deck gets data-transition="<kind>",
// all until INTRO_MS has passed. Kinds are styled in style.css ("title", "morph", "roll", "thanks").
// In a "morph" entrance, each [data-morph="x"] element glides from where the previous slide's
// [data-morph="x"] element was (position and font size) into its own place, while the old one is hidden.
// A slide with data-skip is left out of the deck (and the counter) but kept in the HTML: delete the attribute to bring it back.

const deck = document.querySelector(".deck");
const slides = [...document.querySelectorAll(".slide:not([data-skip])")];
const counter = document.querySelector(".counter");
const showAll = new URLSearchParams(location.search).has("all");
let current = 0;
let step = 0;
const INTRO_MS = 1700;
let introTimer;

function fit() {
  const scale = Math.min(window.innerWidth / 1920, window.innerHeight / 1080);
  deck.style.transform = `translate(-50%, -50%) scale(${scale})`;
}

const fragments = (slide) => [...slide.querySelectorAll(".fragment")];
const stepCount = (slide) => Number(slide.dataset.steps) || fragments(slide).length;

function render() {
  const slide = slides[current];
  fragments(slide).forEach((f, i) => f.classList.toggle("visible", i < step));
  for (let i = 1; i <= stepCount(slide); i++) slide.classList.toggle(`s${i}`, i <= step);
  counter.textContent = `${current + 1} / ${slides.length}`;
  history.replaceState(null, "", `${location.search}#${current + 1}${step ? `.${step}` : ""}`);
}

function clearIntro() {
  clearTimeout(introTimer);
  slides.forEach((s) => s.classList.remove("intro", "outro"));
  document.querySelectorAll(".morph-src").forEach((el) => el.classList.remove("morph-src"));
  delete deck.dataset.transition;
}

function morph(from, to) {
  const scale = deck.getBoundingClientRect().width / 1920;
  to.querySelectorAll("[data-morph]").forEach((el) => {
    const src = from.querySelector(`[data-morph="${el.dataset.morph}"]`);
    if (!src) return;
    src.classList.add("morph-src"); // hidden on the old slide while its copy glides (style.css)
    const a = src.getBoundingClientRect();
    const b = el.getBoundingClientRect();
    const s = parseFloat(getComputedStyle(src).fontSize) / parseFloat(getComputedStyle(el).fontSize);
    const dx = (a.left - b.left) / scale;
    const dy = (a.top + a.height / 2 - b.top - (b.height * s) / 2) / scale;
    el.animate(
      [{ transform: `translate(${dx}px, ${dy}px) scale(${s})`, transformOrigin: "left top" },
       { transform: "none", transformOrigin: "left top" }],
      { duration: 900, delay: 150, easing: "cubic-bezier(0.65, 0, 0.35, 1)", fill: "backwards" },
    );
  });
}

function show(index, atStep = showAll ? Infinity : 0, animate = false) {
  clearIntro();
  const from = slides[current];
  current = Math.max(0, Math.min(slides.length - 1, index));
  step = Math.max(0, Math.min(stepCount(slides[current]), atStep));
  slides.forEach((s, i) => s.classList.toggle("active", i === current));
  const to = slides[current];
  if (animate && to !== from && "intro" in to.dataset && !matchMedia("(prefers-reduced-motion: reduce)").matches) {
    to.classList.add("intro");
    from.classList.add("outro");
    deck.dataset.transition = to.dataset.intro;
    if (to.dataset.intro === "morph") morph(from, to);
    introTimer = setTimeout(clearIntro, INTRO_MS);
  }
  render();
}

function next() {
  if (step < stepCount(slides[current])) { step++; render(); }
  else if (current < slides.length - 1) show(current + 1, 0, true);
}

function prev() {
  if (step > 0 && !showAll) { step--; render(); }
  else if (current > 0) show(current - 1, Infinity);
}

document.addEventListener("keydown", (e) => {
  if (["ArrowRight", "ArrowDown", " ", "PageDown"].includes(e.key)) next();
  else if (["ArrowLeft", "ArrowUp", "PageUp"].includes(e.key)) prev();
  else if (e.key === "Home") show(0);
  else if (e.key === "End") show(slides.length - 1);
  else if (e.key === "f" || e.key === "F") {
    document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen();
  } else return;
  e.preventDefault();
});

// Clicking a link opens it (in a new tab) instead of advancing the slide
document.addEventListener("click", (e) => { if (!e.target.closest("a")) next(); });
window.addEventListener("resize", fit);

fit();
const [slideNo, stepNo] = location.hash.slice(1).split(".").map((n) => parseInt(n, 10));
show((slideNo || 1) - 1, showAll ? Infinity : stepNo || 0);
