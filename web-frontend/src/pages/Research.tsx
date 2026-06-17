import { useMemo, useState } from "react";
import type React from "react";
import { useResearchPapers } from "../hooks/useResearch";
import {
  Badge,
  EmptyState,
  ErrorState,
  SectionHeading,
  Spinner,
} from "../components/ui";

const ALL = "All";

export const Research: React.FC = () => {
  const [activeCategory, setActiveCategory] = useState<string>(ALL);
  const { data: papers, isLoading, isError, refetch } = useResearchPapers();

  const categories = useMemo(() => {
    const set = new Set<string>();
    (papers ?? []).forEach((p) => p.category && set.add(p.category));
    return [ALL, ...Array.from(set).sort()];
  }, [papers]);

  const visible = useMemo(() => {
    if (!papers) return [];
    if (activeCategory === ALL) return papers;
    return papers.filter((p) => p.category === activeCategory);
  }, [papers, activeCategory]);

  return (
    <div className="mx-auto max-w-6xl space-y-8 px-6 py-12">
      <SectionHeading
        eyebrow="Research"
        title="Applied research"
        subtitle="Papers and working notes behind the models that power AlphaMind."
      />

      {isLoading && (
        <div className="flex justify-center py-20">
          <Spinner />
        </div>
      )}

      {isError && (
        <ErrorState
          message="We could not load the research library."
          onRetry={() => refetch()}
        />
      )}

      {!isLoading && !isError && (
        <>
          {categories.length > 1 && (
            <div className="flex flex-wrap gap-2">
              {categories.map((cat) => {
                const active = cat === activeCategory;
                return (
                  <button
                    key={cat}
                    type="button"
                    onClick={() => setActiveCategory(cat)}
                    className={`rounded-full px-4 py-1.5 text-sm font-medium transition-colors ${
                      active
                        ? "bg-brand text-white"
                        : "bg-surface-2 text-ink-muted hover:text-ink"
                    }`}
                  >
                    {cat}
                  </button>
                );
              })}
            </div>
          )}

          {visible.length === 0 ? (
            <EmptyState
              title="No papers yet"
              hint="Research in this category will appear here as it is published."
            />
          ) : (
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
              {visible.map((paper) => (
                <article key={paper.id} className="am-card flex flex-col p-6">
                  <div className="flex items-start justify-between gap-3">
                    <h2 className="font-display text-lg font-semibold leading-snug text-ink">
                      {paper.title}
                    </h2>
                    {paper.category && (
                      <Badge tone="brand">{paper.category}</Badge>
                    )}
                  </div>

                  <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-sm text-ink-muted">
                    {paper.authors?.length > 0 && (
                      <span>{paper.authors.join(", ")}</span>
                    )}
                    {paper.year ? (
                      <span className="font-mono tabular-nums">
                        {paper.year}
                      </span>
                    ) : null}
                  </div>

                  <p className="mt-3 flex-1 text-sm leading-relaxed text-ink-muted">
                    {paper.abstract}
                  </p>

                  {paper.url ? (
                    <a
                      href={paper.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="am-btn am-btn-ghost mt-4 self-start"
                    >
                      Read paper
                      <svg
                        className="ml-1.5 h-4 w-4"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                        aria-hidden="true"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M14 5h5m0 0v5m0-5L10 14M5 7v12h12"
                        />
                      </svg>
                    </a>
                  ) : (
                    <span className="mt-4 self-start text-xs font-medium uppercase tracking-wide text-ink-faint">
                      Preprint in preparation
                    </span>
                  )}
                </article>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default Research;
