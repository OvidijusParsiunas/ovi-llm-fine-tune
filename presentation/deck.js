// Minimal slide engine.
// Next: → ↓ Space PageDown, click.  Previous: ← ↑ PageUp.  Home/End jump.  F toggles fullscreen.
// The URL hash (#3) remembers the current slide, so a refresh keeps your place.
// Elements with class="fragment" appear one at a time as you press "next" on that slide.
// Add ?all to the URL to show every fragment at once (handy for previewing/printing).

const deck = document.querySelector(".deck");
const slides = [...document.querySelectorAll(".slide")];
const counter = document.querySelector(".counter");
const showAll = new URLSearchParams(location.search).has("all");
let current = 0;

function fit() {
  const scale = Math.min(window.innerWidth / 1920, window.innerHeight / 1080);
  deck.style.transform = `translate(-50%, -50%) scale(${scale})`;
}

const fragments = (slide) => [...slide.querySelectorAll(".fragment")];

function show(index, { revealAll = showAll } = {}) {
  current = Math.max(0, Math.min(slides.length - 1, index));
  slides.forEach((s, i) => s.classList.toggle("active", i === current));
  fragments(slides[current]).forEach((f) => f.classList.toggle("visible", revealAll));
  counter.textContent = `${current + 1} / ${slides.length}`;
  history.replaceState(null, "", `${location.search}#${current + 1}`);
}

function next() {
  const hidden = fragments(slides[current]).find((f) => !f.classList.contains("visible"));
  if (hidden) hidden.classList.add("visible");
  else if (current < slides.length - 1) show(current + 1);
}

function prev() {
  const shown = fragments(slides[current]).filter((f) => f.classList.contains("visible"));
  if (shown.length && !showAll) shown.at(-1).classList.remove("visible");
  else if (current > 0) show(current - 1, { revealAll: true });
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

document.addEventListener("click", next);
window.addEventListener("resize", fit);

fit();
show((parseInt(location.hash.slice(1), 10) || 1) - 1);
