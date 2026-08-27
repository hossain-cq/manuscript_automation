# Literature API research notes

## OpenAlex
Source: https://help.openalex.org/api/

- OpenAlex provides a REST API over works, authors, sources, institutions, topics, and related scholarly entities.
- The works endpoint supports search, filtering, sorting, paging, field selection, and single-entity retrieval.
- OpenAlex records expose stable IDs and may include DOI identifiers and abstract information.
- Search can cover titles, abstracts, and other indexed text; the system should use search for candidate discovery and stable IDs/DOIs for normalization.
- The API can be used without a key for initial use; rate limits and authentication should be handled by a provider adapter.
- External text fields should be treated as untrusted input and sanitized/escaped before display or prompt inclusion.

## Crossref
Source: https://www.crossref.org/documentation/retrieve-metadata/rest-api/

- Crossref REST exposes bibliographic metadata deposited by members and trusted sources.
- The /works endpoint supports search and retrieval of scholarly content metadata; /works/{doi} retrieves a single DOI record.
- Records may include titles, authors, publication dates, journals, DOI, licenses, funding, ORCID/ROR identifiers, post-publication data, and sometimes abstracts.
- Crossref metadata is deposited by publishers/members and should be treated as provider metadata, not as proof that a paper supports a candidate claim.
- The system should cache provider responses, preserve retrieval timestamps, store source URLs, and verify DOI/metadata consistency across providers when possible.

Implementation conclusions:
- Use OpenAlex for broad candidate discovery and citation/works graph context.
- Use Crossref for DOI and publisher metadata verification.
- Optionally add Semantic Scholar as a third adapter for citation graph and abstract retrieval, but keep provider-specific records separate until normalized.
- Novelty comparison must use verified literature records plus claim-level comparison; absence of a retrieved similar paper is never evidence of novelty.
