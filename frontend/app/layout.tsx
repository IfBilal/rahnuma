import type { Metadata } from "next";
import { ClerkProvider } from "@clerk/nextjs";
import "./globals.css";

export const metadata: Metadata = {
  title: "Rahnuma | Admissions Advisor",
  description: "A source-grounded Pakistani university admissions advisor.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <ClerkProvider appearance={{ variables: { colorPrimary: "#d7ff3f", colorBackground: "#0d1413", colorForeground: "#edf5ed", colorInputBackground: "#111c1a", colorInputText: "#edf5ed", colorNeutral: "#8e9d96", colorModalBackdrop: "rgba(3, 7, 6, 0.58)", fontFamily: "DM Sans, Arial, sans-serif", fontFamilyButtons: "DM Sans, Arial, sans-serif" } }}><html lang="en"><body>{children}</body></html></ClerkProvider>;
}
