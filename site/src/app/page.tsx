export default function Home() {
  return (
    <main className="wrap">
      <section className="hero">
        <span className="badge">Phase 0 — powering Einstein Labs products first</span>
        <h1>Decide which compute to run.<br /><em>Then run it efficiently.</em></h1>
        <p>
          Serving engines optimize execution and treat the model as fixed.
          Routing products optimize selection and treat models as external APIs.
          Photon is one engine that owns both halves — selection and execution
          co-designed, over fine-tuned open models, on infrastructure you control.
        </p>
      </section>

      <section className="cards">
        <div className="card">
          <h3>Photon Ops — shipping</h3>
          <p>The operational layer: OpenAI-compatible gateway, fine-tuning
          pipelines, model registry with eval gates, per-tenant cost
          attribution, and deployment templates from EKS to air-gapped.</p>
        </div>
        <div className="card">
          <h3>Photon Core — the engine</h3>
          <p>A fleet fabric that serves many bases and hundreds of adapters on
          shared GPUs, and a learned router that sends every request to the
          cheapest compute that meets the quality bar.</p>
        </div>
        <div className="card">
          <h3>Honest by construction</h3>
          <p>The benchmark precedes the claim: measured baselines, golden-set
          quality gates on every routing change, and shadow studies before we
          scale any bet.</p>
        </div>
      </section>

      <section>
        <h2 className="display">Built for deployments, not demos</h2>
        <p style={{ color: "var(--dim)", lineHeight: 1.6, maxWidth: "62ch" }}>
          Photon runs Einstein Labs&apos; own products in production and deploys
          into enterprise VPCs — cloud or on-prem, your models, your data. If
          you need vertical AI systems running on infrastructure you own, that
          is exactly the engagement Photon exists for.
        </p>
        <p style={{ marginTop: 26 }}>
          <a className="cta" href="https://einsteinlabz.com">Talk to us about enterprise deployments</a>
        </p>
      </section>
    </main>
  );
}
