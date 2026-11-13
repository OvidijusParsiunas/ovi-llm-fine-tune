// Minimal slide engine.
// Next: → ↓ Space PageDown, click.  Previous: ← ↑ PageUp.  Home/End jump.  F toggles fullscreen.
// The URL hash (#3) remembers the current slide, so a refresh keeps your place.

const deck = document.querySelector(".deck");
const slides = [...document.querySelectorAll(".slide")];
const counter = document.querySelector(".counter");
let current = 0;

function fit() {
  const scale = Math.min(window.innerWidth / 1920, window.innerHeight / 1080);
  deck.style.transform = `translate(-50%, -50%) scale(${scale})`;
}

function show(index) {
  current = Math.max(0, Math.min(slides.length - 1, index));
  slides.forEach((s, i) => s.classList.toggle("active", i === current));
  counter.textContent = `${current + 1} / ${slides.length}`;
  history.replaceState(null, "", `#${current + 1}`);
}

document.addEventListener("keydown", (e) => {
  if (["ArrowRight", "ArrowDown", " ", "PageDown"].includes(e.key)) show(current + 1);
  else if (["ArrowLeft", "ArrowUp", "PageUp"].includes(e.key)) show(current - 1);
  else if (e.key === "Home") show(0);
  else if (e.key === "End") show(slides.length - 1);
  else if (e.key === "f" || e.key === "F") {
    document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen();
  } else return;
  e.preventDefault();
});

document.addEventListener("click", () => show(current + 1));
window.addEventListener("resize", fit);

fit();
show((parseInt(location.hash.slice(1), 10) || 1) - 1);
