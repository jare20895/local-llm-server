import "swagger-ui-react/swagger-ui.css";
import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Vector-Tester | LLM Model Testing",
  description:
    "Companion tester UI for orchestrating high-signal LLM load tests and capturing telemetry.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <div className="layout">{children}</div>
      </body>
    </html>
  );
}
