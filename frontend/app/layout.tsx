import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Rahnuma | Admissions Advisor",
  description: "A source-grounded Pakistani university admissions advisor.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
