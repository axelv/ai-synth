# The public site

One landing page and one page per instrument, published to GitHub Pages.

```
python site/build.py                        # everything, into site/dist/
python site/build.py --patch acid-line      # one instrument, for a quick loop
python -m http.server 8791 --directory site/dist
```

`catalogue.json` says which instruments exist and what each one is. `config.json` holds
the deployment decisions: where a request goes, whether a counter is installed, whether
the plugin exists yet. Neither is code, and changing what the site says should not mean
touching `build.py`.

## What this page is an experiment about

`VISION.md` describes an app where a musician drops in a clip and gets back a patch.
The expensive half of that has never been demand-tested. This page is the cheapest
instrument that can test it: **does anyone ask for a sound?**

Everything else on the page is in service of getting a stranger to the point where
asking is a reasonable thing to do. That is the whole design brief, and it is why the
page is ordered the way it is.

## The order, and why

**Sound before argument.** The instrument sits above every word explaining it. Browser
autoplay policy makes one click unavoidable, so that click plays a phrase and lights
the keys rather than opening a silent panel: silence after the first click is where a
visitor concludes the page is broken, because nothing on screen says the keys are
playable until something plays them. The hero `Play` is a forwarded click on the
instrument's own button, moved above the panel because the panel's header pushes its
real button near the fold on a phone, which is where most shared links get opened.

**Then three steps, in the order a person meets them.** Press play, take the keys, turn
the knobs. Three, not five, and each names the thing you touch rather than the thing it
does internally. Step three is where the difference between this and a sample lives: it
keeps dialling.

**Then the instruments, named.** A specific instrument in a tab converts better than a
generic "build any patch" promise, and named machines are what search finds and what
people post. Six is thin; eight to twelve is the target.

**Then the ask**, which is the only measurement that matters. It is stated honestly:
built by hand, roughly a day, no service behind it yet. The honesty is not decoration.
A person who knows they are talking to one person forgives a slow reply, and a request
made under that understanding is a truer demand signal than one made to what looks like
a product.

**Then the developer door, deliberately last.** Installing the skill is a different
audience with a different threshold, and putting it above the musician path would cost
the visitor who cannot read Faust and does not want to.

## What is deliberately not on it

- **No signup, no email gate, no cookie banner.** Anything that stands between the
  first click and the first sound is spending the only currency this page has.
- **No pricing, no roadmap, no waitlist.** There is nothing to sell yet, and a waitlist
  measures politeness rather than demand.
- **No account required to hear anything.** Every instrument page is one file that
  fetches nothing at runtime.
- **No claim to be a machine it is not.** Instruments are named for what they sound
  like. The panel homage carries its attribution on the page that uses it.

## What would make the experiment readable

Two numbers, and only two:

1. **Requests submitted.** `config.json`'s `request_form_url` is empty until a form
   endpoint exists, which leaves the GitHub issue as the only path and silently
   excludes every musician without an account. Filling it in is the first thing to do
   after the site is live, because until then the headline number is measuring GitHub
   accounts.
2. **Plays.** `config.json`'s `analytics` is injected verbatim into the head of every
   page and is empty by default. Whatever goes there should not need a consent banner
   in front of the one click the page depends on.

A request log lives in the issue tracker under the `sound request` label, which is
also, for free, the thing issue #8 asks for: what people ask for in their own words is
a set of test cases nobody in this repo would have written.

## Known gaps

- **No `og:image`.** Every platform that renders a social preview wants a raster, and
  this build is stdlib-only by design, so nothing here can draw one. A shared link
  currently previews as text, which costs reach. Fix is a committed PNG.
- **Six instruments, not eight to twelve**, and two of them carry a measured defect
  named on their card. Filling the catalogue with patches built for it, rather than
  borrowed from the skill's teaching corpus, is the next real content work.
- **`site/check.mjs` is not in CI.** It needs a browser, and `npm ci` runs on the
  publishing path.
