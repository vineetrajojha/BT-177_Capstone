import { MetadataRoute } from "next";

export default function sitemap(): MetadataRoute.Sitemap {
  const baseUrl = "https://bevfusionlite.vineetraj.tech";

  const routes = [
    "",
    "/demo",
    "/results",
    "/ablations",
    "/nuscenes",
    "/architecture",
    "/literature",
    "/reproduce",
    "/reference-models",
    "/downloads",
    "/about",
  ];

  return routes.map((route) => ({
    url: `${baseUrl}${route}`,
    lastModified: new Date(),
    changeFrequency: "weekly" as const,
    priority: route === "" ? 1.0 : 0.8,
  }));
}
