# The keyword proxy

API Gateway, then one Lambda, then Pinecone (keywords) and Elasticsearch (papers). It lets anyone read both without
holding a key: the two secrets, a **read-only** Pinecone key and a **read-only** Elasticsearch key, stay in AWS. Callers who want to search a phrase that is not a keyword embed
it themselves with their own OpenAI key and send only the vector.

```
package  ->  https://<id>.execute-api.<region>.amazonaws.com  ->  Lambda  ->  Pinecone       (keywords)
                                                                              ->  Elasticsearch  (papers)
```

| Request | Does |
|---|---|
| `GET /v1/health` | `{"ok": true}` |
| `GET /v1/manifest` | the embedding model and size the vectors were made with, the keyword count, where a downloadable copy lives |
| `POST /v1/neighbors` | `{"name": "firm size", "k": 25, "min_works": 500, "distinct": true}`: the nearest keywords to a **keyword**. A name that is not a keyword gets a 404 |
| `POST /v1/papers` | `{"query": "family firms decentralization", "k": 10}`: papers matching a search (title and abstract), from Elasticsearch. 404 if it is not switched on |
| `POST /v1/query` | `{"vector": [512 numbers], "k": 25, "filter": {"works": {"$gte": 500}}}`: the nearest keywords to a vector you made |
| `POST /v1/fetch` | `{"ids": ["firm-size"]}`: the stored vectors of up to 10 keywords |

Nothing else exists: no writes, no other filters, no catch-all route. Inputs are checked before the database is
asked, and the database's own error messages are logged, never returned.

## From Python

```python
from confoundsearch import KeywordService, OpenAIEmbedder

db = KeywordService("https://<id>.execute-api.<region>.amazonaws.com", embedder=OpenAIEmbedder())  # your key
db.query("firm size")  # a keyword: looked up by the service, no OpenAI call
db.query("how big a firm is")  # not a keyword: embedded with YOUR key here, then searched
```

Or from a terminal: `python devops/tools/keyword_vectors.py query "firm size" --api https://<address>`.

From the agent: set `CONFOUNDSEARCH_API=https://<address>` in `.env` (or pass `--api`). The default `--corpus all`
then searches the service together with Semantic Scholar and OpenAlex (and the hosted index too, if `ES_URL` is
set), merges the results and removes duplicates. A source that is down is skipped, and the status line says
which; the run only stops if every source fails. `--corpus service` searches the service alone.

## Deploying it

**Not yet done from this repository: nothing here has been deployed or run against AWS.** The handler and the
client are tested against fakes; the template is checked for shape, and has not been validated by CloudFormation.
Treat the first deploy as a test and read what it creates.

You need an AWS account and, if you can get one, a **read-only** Pinecone API key (never the key you write with).
Choose one AWS **region** and use it for every step below (`us-east-1` is fine); the S3 bucket, the two parameters
and the stack must all be in it.

### First, for either route

1. **Build the zip** (our files only; no dependencies): `just proxy-build` makes `devops/proxy/build/lambda.zip`.
2. **Put it in S3.** Make a bucket (any unique name, say `confoundsearch-deploy-<something>`) and upload the zip.
   Name the uploaded file for its version, for example `lambda-2026-11-02.zip`. The function updates only when the
   name changes, so use a new name each time you deploy new code.
3. **Put the keys in AWS**, encrypted (Systems Manager, Parameter Store, Create parameter; type **SecureString**):
   - `/keywords/pinecone-read-key`: the Pinecone key
   - `/keywords/elastic-read-key`: an Elasticsearch API key limited to **read on the papers index** (Kibana: Stack
     Management, API keys; a role with `read` and `view_index_metadata` on `papers` only). Skip it, and leave
     `ElasticUrl` empty, to run keywords only.
4. **Note your hosted zone ID** (Route 53, Hosted zones, `confoundsearch.org`; the `Z...` value).

### In the console (nothing to install)

CloudFormation, **Create stack**, **With new resources (standard)**, **Upload a template file**, choose
`devops/proxy/template.yaml`. On the next page fill in the form (the parameters are listed below). The ones you
must set:

| Parameter | Value |
|---|---|
| Stack name | `confoundsearch-service` |
| `CodeBucket`, `CodeKey` | the bucket and the file name from step 2 |
| `PineconeIndex`, `PineconeHost` | your index name, and its host from the Pinecone console (without `https://`) |
| `ElasticUrl` | your Elasticsearch endpoint, or empty |
| `DomainName`, `HostedZoneId` | `confoundsearch.org`, and the zone ID from step 4 |
| `AlertEmail` | where a spend warning should go (optional) |

Leave the rest as they are. On the last page tick the boxes that acknowledge it will create IAM resources and
that it uses a transform (`AWS::Serverless`), then **Submit**. CloudFormation shows each resource as it is
made; the certificate can take several minutes to validate. If the stack ends in `ROLLBACK_COMPLETE`, open its
**Events** tab: the first red line says why. Delete the failed stack and try again.

When it says `CREATE_COMPLETE`, the **Outputs** tab shows `PermanentUrl`.

### Or on the command line

With the AWS CLI (no SAM needed):
```bash
aws cloudformation deploy --template-file devops/proxy/template.yaml --stack-name confoundsearch-service \
  --capabilities CAPABILITY_IAM CAPABILITY_AUTO_EXPAND --region us-east-1 --parameter-overrides \
  CodeBucket=<bucket> CodeKey=<file name> PineconeIndex=<index> PineconeHost=<host> \
  ElasticUrl=<url> DomainName=confoundsearch.org HostedZoneId=<Z...>
```
(or `sam deploy`, which accepts the same parameters).

### Check it

```bash
curl https://confoundsearch.org/v1/health
curl https://confoundsearch.org/v1/manifest
curl -X POST https://confoundsearch.org/v1/neighbors -d '{"name": "firm size", "k": 5}'
just check-domain confoundsearch.org
```
(Without a domain, use the stack's `ApiUrl` output instead.)

### Updating the code later

Build, upload the new zip under a **new name**, then update the stack (CloudFormation, the stack, **Update**, *Use
existing template*) with the new `CodeKey`.

### Settings (template parameters)

| Parameter | Default | Meaning |
|---|---|---|
| `CodeBucket`, `CodeKey` | none: asked every time | the S3 bucket and file name of the zip |
| `PineconeKeyParameter` | `/keywords/pinecone-read-key` | where the key is stored in AWS |
| `PineconeIndex`, `PineconeHost` | `keywords`, empty | the index to read |
| `ElasticUrl`, `ElasticKeyParameter`, `ElasticIndex` | empty, `/keywords/elastic-read-key`, `papers` | paper search. An empty `ElasticUrl` switches it off |
| `EmbedModel`, `EmbedDims` | `text-embedding-3-small`, `512` | what the vectors were built with; shown in the manifest. The size is checked on every `/v1/query` (a vector of any other length is refused) |
| `RateLimit`, `BurstLimit` | `5`, `10` | requests per second the whole API accepts. Above that, API Gateway answers 429 |
| `MaxConcurrency` | `5` | at most this many copies of the function at once: a hard cap on what a flood can cost |
| `AlertEmail`, `MonthlyBudgetUsd` | empty, `10` | a warning at 80% of the month's budget. Empty skips it |
| `KeywordCount`, `LocalCopyUrl` | empty | shown in the manifest |

### What protects the bill

- **Throttling** on the API (rate and burst), and **a cap on concurrent function runs**.
- **A budget alert**, if you give an email address.
- **No OpenAI key on the server**, so there is no model bill to run up. A query costs Pinecone read units, which count
  against your Pinecone plan; set a usage alert there too. A paper search costs Elasticsearch search time on your
  project; answers are capped at 50 papers and queries at 300 characters.
- If the address leaks and is abused, delete the stack (`sam delete`) or lower `RateLimit`. The key is read-only
  and can be rotated in the Pinecone console and in `aws ssm put-parameter --overwrite`.

### A permanent address (your own domain)

The address API Gateway generates (`https://<id>.execute-api...`) changes if the stack is ever deleted and
recreated. A domain you own does not: the package carries `https://<your domain>`, and you can point that name
at any deployment, now or in five years. The template can set it all up. It makes nothing unless you give it a
`DomainName`.

You need a domain name. Register one with any registrar (a plain, neutral name is best for an anonymous review;
ask for WHOIS privacy so that the registration does not carry your name). Then:

**If its DNS is in Route 53** (the simple case): give `DomainName=<your domain>` and its `HostedZoneId` (Route 53 console, Hosted zones, the `Z...` ID). The
stack requests a certificate, validates it by DNS on its own, makes the API Gateway domain and the DNS record.
```bash
# in the console form, or on the command line as a --parameter-overrides value:
DomainName=confoundsearch.org HostedZoneId=Z0123456789ABC
```
**If its DNS is somewhere else**: request a certificate for the name in AWS Certificate Manager *in the same region as
the stack*, validate it (ACM tells you the CNAME record to add), and give its ARN as `CertificateArn`. After the
deploy, add a CNAME from your name to the `DnsTarget` the stack prints.

Either way the stack prints `PermanentUrl`. That is the address to use; check it first:
```bash
curl https://confoundsearch.org/v1/health
```
Once it works, you can close the generated address by redeploying with `DisableDefaultAddress=true`, so that the domain
is the only way in. (The template ignores that setting if there is no domain, so it cannot lock you out of everything.)

What it costs, from memory: the certificate is free; a Route 53 hosted zone is about $0.50 a month; the domain is a
yearly fee that depends on the name. Check the current prices.

The service answers on the bare domain, with every route under `/v1/`. Route 53 can point a bare domain at API
Gateway (an alias record, which is what the template makes). Two consequences: a registrar or DNS host other than
Route 53 usually cannot (a CNAME is not allowed at the bare domain), and the API owns the whole name, so a website
at `https://confoundsearch.org/` would have to move to a subdomain such as `www`. Anything under `/` that is not a
`/v1/` route answers 404.

This part of the template has not been run against AWS either. The first deploy with a domain is a test: read the
change set before you approve it.

### Making it the default

The package can carry the address, so that nobody configures anything. After the first successful deploy, put the
`PermanentUrl` that the stack prints (use `ApiUrl` only if you did not set up a domain) into one line of
`confoundsearch/data/keyword_service.py`:

```python
DEFAULT_SERVICE_URL = "https://confoundsearch.org"
```

From then on `--corpus all` (the default) and `KeywordService()` use it with no setting. Things to decide first:

- **Every user's search terms go to your server by default.** Say so in the README. Anyone can switch it off with
  `--api off` (or `CONFOUNDSEARCH_API=off`), and can point at their own deployment with `--api URL`.
- **Put your own domain in the code, not the generated address.** The generated one changes if the stack is ever
  recreated, and old installs would stop reaching it (they carry on with the other sources, since a service that
  is down is skipped). The domain stays yours.
- **The address is public** once it is in the repository. That is the point, and why the key behind it is read-only,
  the API is throttled, and concurrency is capped.

Until `DEFAULT_SERVICE_URL` is set it is empty, the service is simply not part of the default, and nothing changes.

### Not done yet

- The fallback to a downloadable local copy when the service is down (the manifest has a place for its address).
- Per-caller limits: throttling is for the whole API, not per user.
