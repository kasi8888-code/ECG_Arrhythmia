import { Inter } from "next/font/google";
import "./globals.css";
import Navigation from "@/components/Navigation";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

export const metadata = {
  title: "ECG Arrhythmia Detection System",
  description: "AI-assisted cardiac rhythm analysis powered by deep learning",
  keywords: ["ECG", "Arrhythmia", "Detection", "AI", "Deep Learning", "Healthcare"],
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className={`${inter.variable} font-sans antialiased bg-gray-50 min-h-screen`}>
        <Navigation />
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {children}
        </main>

        {/* Footer */}
        <footer className="border-t border-gray-200 bg-white mt-auto">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
            <div className="flex flex-col md:flex-row justify-between items-center text-sm text-gray-500">
              <p>© 2026 ECG Arrhythmia Detection System. For research and educational purposes.</p>
              <p className="mt-2 md:mt-0">
                ⚠️ Not a substitute for professional medical diagnosis.
              </p>
            </div>
          </div>
        </footer>
      </body>
    </html>
  );
}
