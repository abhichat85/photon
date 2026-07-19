import { notFound } from "next/navigation";
import { fetchPost } from "@/lib/prism";

export default async function Post({ params }: { params: { slug: string } }) {
  const post = await fetchPost(params.slug);
  if (!post || !post.html) notFound();
  return (
    <main className="wrap article">
      <h1>{post.title}</h1>
      <div className="meta">
        {new Date(post.published_at).toDateString()} · {post.author}
      </div>
      <div className="article-body" dangerouslySetInnerHTML={{ __html: post.html }} />
    </main>
  );
}
