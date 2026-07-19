import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Photon — The routed inference engine",
  description: "Decide which compute to run, then run it efficiently. Inference operations for Einstein Labs products and enterprise deployments.",
  icons: { icon: "/brand/favicon.svg", apple: "/brand/apple-touch-icon.png" },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        {/* eslint-disable-next-line @next/next/no-css-tags */}
        <link rel="stylesheet" href="/brand/brand.css" />
      </head>
      <body>
        <header className="site-header">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <Link href="/"><img src="/brand/logo.svg" alt="Photon" height={32} /></Link>
          <nav><Link href="/blog">Blog</Link></nav>
        </header>
        {children}
        <footer className="site-footer">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/brand/einstein-mark.svg" alt="" width={18} height={18} />
          <span>An <a href="https://einsteinlabz.com">Einstein Labs</a> product</span>
        </footer>
      </body>
    </html>
  );
}
