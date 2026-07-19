import Link from "next/link";
import { fetchPosts } from "@/lib/prism";

export const metadata = { title: "Photon — Blog" };

export default async function Blog() {
  const posts = await fetchPosts(50);
  return (
    <main className="wrap">
      <section className="hero" style={{ paddingBottom: 10 }}>
        <h1>Essays</h1>
        <p>Notes on what we are building and why.</p>
      </section>
      {posts.length === 0 ? (
        <div className="empty">Essays coming soon.</div>
      ) : (
        <ul className="post-list">
          {posts.map((p) => (
            <li key={p.slug}>
              <div className="meta">{new Date(p.published_at).toDateString()}</div>
              <h3><Link href={`/blog/${p.slug}`}>{p.title}</Link></h3>
              <p>{p.excerpt}</p>
            </li>
          ))}
        </ul>
      )}
    </main>
  );
}
