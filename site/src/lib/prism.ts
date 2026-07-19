const PRISM_URL = process.env.PRISM_URL ?? "http://localhost:3470";
const PRODUCT = "photon";

export interface PrismPost {
  slug: string; title: string; excerpt: string; product: string;
  tags: string[]; cover: string | null; published_at: string; author: string;
  html?: string;
}

async function get(path: string): Promise<unknown | null> {
  try {
    const ctl = new AbortController();
    const t = setTimeout(() => ctl.abort(), 5000);
    const res = await fetch(`${PRISM_URL}${path}`, {
      signal: ctl.signal, next: { revalidate: 300 },
    });
    clearTimeout(t);
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null; // Prism unreachable -> graceful empty state
  }
}

export async function fetchPosts(limit = 20): Promise<PrismPost[]> {
  const data = (await get(`/api/posts?product=${PRODUCT}&limit=${limit}`)) as
    { posts?: PrismPost[] } | null;
  return data?.posts ?? [];
}

export async function fetchPost(slug: string): Promise<PrismPost | null> {
  return (await get(`/api/posts/${encodeURIComponent(slug)}`)) as PrismPost | null;
}
